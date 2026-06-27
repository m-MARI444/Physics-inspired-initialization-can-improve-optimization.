"""
thermodynamics.py
=================
PSSA Thermodynamic Lifecycle Controllers — Phase 4 Thermodynamic Control

Implements three cooperating systems:
  1. MuonOptimizer      — Newton-Schulz orthogonal gradient optimizer (k*=1 routing)
  2. TransitionAccelerator — Rolling Gram matrix sentinel; detects and forces grokking
  3. EquilibriumLock    — Captures sparse basis post-grokking; prevents re-densification
  4. SpectralSparsityEnforcer - Phase 4 spectral homeostasis

All three are designed to be checkpoint-safe: full state serializes to/from a plain dict
so thermodynamic state survives laptop reboots mid-campaign.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. MUON OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2, "Newton-Schulz requires 2D matrices"

    g_norm = G.norm()
    # Guard: if gradient is NaN/inf (can happen with fp16 overflow), return zeros
    if not torch.isfinite(g_norm) or g_norm < 1e-12:
        return torch.zeros_like(G)

    X = G / (g_norm + 1e-8)

    if G.shape[0] < G.shape[1]:
        for _ in range(steps):
            A = X @ X.T
            X = (1.5 * X) - (0.5 * A @ X)
            if not torch.isfinite(X).all():
                return torch.zeros_like(G)
    else:
        for _ in range(steps):
            A = X.T @ X
            X = (1.5 * X) - (0.5 * X @ A)
            if not torch.isfinite(X).all():
                return torch.zeros_like(G)

    return X


class MuonOptimizer:
    def __init__(
        self,
        routing_params: List[Tuple[str, nn.Parameter]],
        other_params:   List[Tuple[str, nn.Parameter]],
        lr_muon:     float = 0.02,
        lr_adam:     float = 1e-3,
        weight_decay: float = 1e-2,
        ns_steps:    int   = 5,
        betas:       Tuple = (0.9, 0.95),
    ):
        self.routing_params  = routing_params
        self.other_params    = other_params
        self.lr_muon         = lr_muon
        self.lr_adam         = lr_adam
        self.weight_decay    = weight_decay
        self.ns_steps        = ns_steps
        self.betas           = betas
        self.step_count      = 0

        self._adam_state: Dict[str, Dict] = {}
        for name, p in other_params:
            # Store on the same device as parameter (GPU) to eliminate CPU-GPU sync bottlenecks.
            self._adam_state[name] = {
                "m": torch.zeros_like(p.data),
                "v": torch.zeros_like(p.data),
                "delta": torch.zeros_like(p.data),
            }

    def zero_grad(self):
        for _, p in self.routing_params + self.other_params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_count += 1
        t = self.step_count
        b1, b2 = self.betas
        eps = 1e-8

        # ── Muon update for routing matrices ──────────────────────────────
        for name, p in self.routing_params:
            if p.grad is None:
                continue
            G = p.grad.data

            if G.ndim == 2:
                G_orth = _newton_schulz_orthogonalize(G, steps=self.ns_steps)
            else:
                G_orth = G / (G.norm() + 1e-8)

            p.data.mul_(1.0 - self.lr_muon * self.weight_decay)
            p.data.add_(G_orth, alpha=-self.lr_muon)

        # ── AdamW update for world model + projections (fully on GPU) ─────
        # Performs momentum arithmetic fully on GPU to avoid CPU-GPU sync stalls.
        import math
        bias_correction = math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        for name, p in self.other_params:
            if p.grad is None:
                continue

            G = p.grad.data.float()
            state = self._adam_state[name]

            # In-place momentum updates on GPU
            state["m"].mul_(b1).add_(G, alpha=1 - b1)
            state["v"].mul_(b2).addcmul_(G, G, value=1 - b2)

            # In-place step computation on GPU
            torch.sqrt(state["v"], out=state["delta"])
            state["delta"].add_(eps)

            # delta = m / delta
            state["delta"].reciprocal_().mul_(state["m"])

            # Apply bias correction and clamp update on GPU
            state["delta"].mul_(bias_correction).clamp_(-0.1, 0.1)

            # Apply weight decay + update directly on GPU
            p.data.mul_(1.0 - self.lr_adam * self.weight_decay)
            p.data.add_(state["delta"], alpha=-self.lr_adam)

    def state_dict(self) -> dict:
        return {
            "step_count":  self.step_count,
            "adam_state":  {
                name: {k: v.cpu() for k, v in s.items() if k in ["m", "v"]}
                for name, s in self._adam_state.items()
            },
            "lr_muon":     self.lr_muon,
            "lr_adam":     self.lr_adam,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, sd: dict):
        self.step_count  = sd["step_count"]
        self.lr_muon     = sd.get("lr_muon",     self.lr_muon)
        self.lr_adam     = sd.get("lr_adam",     self.lr_adam)
        self.weight_decay = sd.get("weight_decay", self.weight_decay)
        for name, s in sd["adam_state"].items():
            if name in self._adam_state:
                self._adam_state[name]["m"].copy_(s["m"])
                self._adam_state[name]["v"].copy_(s["v"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSITION ACCELERATOR
# ─────────────────────────────────────────────────────────────────────────────

class TransitionAccelerator:
    PHASE_NAMES = {1: "DENSE", 2: "COMPRESSING", 3: "GROKKING", 4: "LOCKED"}

    def __init__(
        self,
        sentinel_params: List[Tuple[str, nn.Parameter]],
        gap_threshold:   float = 0.35,
        window:          int   = 50,
        check_every:     int   = 100,
        wd_boost_factor: float = 3.0,
    ):
        self.sentinel_params  = sentinel_params
        self.gap_threshold    = gap_threshold
        self.window           = window
        self.check_every      = check_every
        self.wd_boost_factor  = wd_boost_factor

        self.phase            = 1
        self.step_count       = 0
        self.gap_history: List[float] = []
        self._grad_buffer     = deque(maxlen=window)
        self._adam_v: Dict[str, torch.Tensor] = {}

    def update_adam_v(self, name: str, v: torch.Tensor):
        self._adam_v[name] = v.detach()

    def _collect_sentinel_grad(self):
        vecs = []
        for _, p in self.sentinel_params:
            if p.grad is not None:
                vecs.append(p.grad.detach().flatten())
        if vecs:
            # Keep on GPU to prevent CPU-GPU synchronization bottlenecks
            self._grad_buffer.append(torch.cat(vecs).bfloat16())

    def _compute_spectral_gap(self) -> float:
        if len(self._grad_buffer) < self.window:
            return 0.0

        # Move to CPU only when performing the check (every 100 steps)
        U = torch.stack(list(self._grad_buffer)).float().cpu()  # [W, d]
        G = U @ U.T                                # [W, W]

        try:
            eigvals = torch.linalg.eigvalsh(G)
        except Exception:
            return 0.0

        eigvals = eigvals.flip(0)  # descending
        if eigvals[0].abs() < 1e-10:
            return 0.0

        gap = (eigvals[0] - eigvals[1]).item() / (eigvals[0].item() + 1e-8)
        return max(gap, 0.0)

    def _apply_selective_wd_boost(self, base_wd: float):
        for name, p in self.sentinel_params:
            if name not in self._adam_v or p.grad is None:
                continue
            v = self._adam_v[name]
            v_mean = v.mean().item()
            v_max  = v.max().item() + 1e-8
            density = v_mean / v_max

            effective_wd = base_wd * (1.0 + (self.wd_boost_factor - 1.0) * density)

            with torch.no_grad():
                p.data.mul_(1.0 - effective_wd * 0.02)

    def step(
        self,
        base_wd: float = 1e-2,
        verbose: bool  = True,
    ) -> int:
        self.step_count += 1
        self._collect_sentinel_grad()

        if self.step_count % self.check_every != 0:
            return self.phase

        gap = self._compute_spectral_gap()
        self.gap_history.append(gap)

        if verbose:
            print(
                f"[TA step {self.step_count:6d}] "
                f"phase={self.PHASE_NAMES[self.phase]} "
                f"gap={gap:.4f} "
                f"(threshold={self.gap_threshold:.2f})",
                flush=True,
            )

        if self.phase == 1 and gap > 0.15:
            self.phase = 2
            if verbose:
                print(f"[TA] >>> Phase 2: spectral compression beginning (gap={gap:.4f})", flush=True)

        elif self.phase == 2:
            self._apply_selective_wd_boost(base_wd)
            recent_gaps = self.gap_history[-3:]
            if len(recent_gaps) >= 2 and all(g > self.gap_threshold for g in recent_gaps):
                self.phase = 3
                if verbose:
                    print(
                        f"[TA] >>> Phase 3: GROKKING TRANSITION DETECTED "
                        f"(gap={gap:.4f} > {self.gap_threshold:.2f} for {len(recent_gaps)} checks)",
                        flush=True,
                    )

        return self.phase

    def state_dict(self) -> dict:
        return {
            "phase":       self.phase,
            "step_count":  self.step_count,
            "gap_history": self.gap_history,
        }

    def load_state_dict(self, sd: dict):
        self.phase      = sd["phase"]
        self.step_count = sd["step_count"]
        self.gap_history = sd.get("gap_history", [])
        print(
            f"[TA] Resumed at step {self.step_count}, "
            f"phase={self.PHASE_NAMES[self.phase]}",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. EQUILIBRIUM LOCK
# ─────────────────────────────────────────────────────────────────────────────

class EquilibriumLock:
    def __init__(
        self,
        routing_params:    List[Tuple[str, nn.Parameter]],
        worldmodel_params: List[Tuple[str, nn.Parameter]],
        lock_strength: float = 0.1,
        clamp_value:   float = 10.0,
        ns_steps:      int   = 5,
    ):
        self.routing_params    = routing_params
        self.worldmodel_params = worldmodel_params
        self.lock_strength     = lock_strength
        self.clamp_value       = clamp_value
        self.ns_steps          = ns_steps

        self._k_star: Dict[str, int] = {}
        self._locked_bases: Dict[str, torch.Tensor] = {}
        self.basis_captured = False
        self.capture_step   = -1

    def capture_basis(self, step: int, verbose: bool = True):
        if self.basis_captured:
            return

        if verbose:
            print(f"[EL] Capturing sparse basis at step {step}...", flush=True)

        all_params = (
            [(name, p, 1) for name, p in self.routing_params] +
            [(name, p, 2) for name, p in self.worldmodel_params]
        )

        for name, p, group_type in all_params:
            if p.dim() < 2:
                continue
            W = p.data.float()
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                if verbose:
                    print(f"[EL] SVD failed for {name}: {e}", flush=True)
                continue

            # Dynamically select k* using singular value energy threshold
            total_energy = S.sum()
            cum_energy = torch.cumsum(S, dim=0)
            target_ratio = 0.85 if group_type == 1 else 0.95
            
            k_star = 1
            for idx in range(len(S)):
                if (cum_energy[idx] / (total_energy + 1e-8)).item() >= target_ratio:
                    k_star = idx + 1
                    break
            
            # Enforce limits (clamping) to prevent memory expansion while ensuring representational degrees of freedom
            max_k = 4 if group_type == 1 else 8
            k_star = max(1, min(k_star, max_k))

            self._locked_bases[name] = U[:, :k_star].to(p.device).clone()
            self._k_star[name]       = k_star

            if verbose:
                sv_ratio = (S[:k_star].sum() / S.sum()).item()
                print(
                    f"[EL]   {name:40s} k*={k_star} "
                    f"top-{k_star} SV energy={sv_ratio:.3f} (target={target_ratio:.2f})",
                    flush=True,
                )

        self.basis_captured = True
        self.capture_step   = step
        if verbose:
            print(
                f"[EL] Basis captured. {len(self._locked_bases)} matrices locked.",
                flush=True,
            )

    def _get_lock_pairs(self, device):
        if not hasattr(self, '_lock_pairs') or self._lock_pairs_device != device or len(self._lock_pairs) != len(self._locked_bases):
            self._lock_pairs = []
            all_params = self.routing_params + self.worldmodel_params
            for name, p in all_params:
                if name in self._locked_bases and p.dim() >= 2:
                    # Keep basis on correct device/dtype once
                    U = self._locked_bases[name].to(device, dtype=p.dtype, non_blocking=True)
                    self._lock_pairs.append((p, U))
            self._lock_pairs_device = device
        return self._lock_pairs

    def lock_loss(self) -> torch.Tensor:
        if not self.basis_captured:
            return torch.tensor(0.0)

        device = next(
            (p.device for _, p in self.routing_params + self.worldmodel_params),
            torch.device("cpu"),
        )
        total = torch.tensor(0.0, device=device)
        pairs = self._get_lock_pairs(device)

        for p, U in pairs:
            # Memory-optimized orthonormal projection loss:
            # ||p - U @ U.T @ p||^2 = ||p||^2 - ||U.T @ p||^2 (since U is orthonormal).
            # This avoids allocating intermediate [d_out, d_in] projection matrices, saving ~400MB VRAM.
            coeff = U.T @ p  # [k*, d_in] (extremely small matrix, k* <= 2)
            p_var = p.pow(2).mean()
            c_var = coeff.pow(2).sum() / p.numel()
            total = total + (p_var - c_var)

        if len(pairs) > 0:
            total = total / len(pairs)
        return self.lock_strength * total

    def clamp_activations(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, -self.clamp_value, self.clamp_value)

    def spectral_health_check(self, verbose: bool = True) -> bool:
        if not self.basis_captured:
            return True

        all_healthy = True
        all_params  = self.routing_params + self.worldmodel_params

        for name, p in all_params:
            if name not in self._k_star or p.dim() < 2:
                continue

            k_star    = self._k_star[name]
            S         = torch.linalg.svdvals(p.data.float())
            active    = (S > 0.01 * S[0]).sum().item()
            stable_r  = (torch.norm(p.data, "fro") ** 2 / (S[0] ** 2 + 1e-8)).item()
            orth_frac = 0.0

            if name in self._locked_bases:
                U_locked  = self._locked_bases[name].to(p.device, p.dtype)
                W_proj    = U_locked @ (U_locked.T @ p)
                W_orth    = p - W_proj
                orth_frac = (torch.norm(W_orth, "fro") / (torch.norm(p, "fro") + 1e-8)).item()

            status = "OK"
            if active > k_star * 4:
                status = "WARN:re-densifying"
                all_healthy = False
            if orth_frac > 0.25:
                status = f"WARN:orth_frac={orth_frac:.3f}"
                all_healthy = False

            if verbose:
                print(
                    f"[EL health] {name:40s} "
                    f"k*={k_star} active={int(active):3d} "
                    f"stable_rank={stable_r:.2f} "
                    f"orth_frac={orth_frac:.3f} "
                    f"[{status}]",
                    flush=True,
                )

        return all_healthy

    def state_dict(self) -> dict:
        return {
            "basis_captured": self.basis_captured,
            "capture_step":   self.capture_step,
            "k_star":         self._k_star,
            "locked_bases":   {
                name: basis.cpu().float()
                for name, basis in self._locked_bases.items()
            },
            "lock_strength":  self.lock_strength,
            "clamp_value":    self.clamp_value,
        }

    def load_state_dict(self, sd: dict, device: torch.device = None):
        self.basis_captured  = sd["basis_captured"]
        self.capture_step    = sd["capture_step"]
        self._k_star         = sd["k_star"]
        self.lock_strength   = sd.get("lock_strength", self.lock_strength)
        self.clamp_value     = sd.get("clamp_value",   self.clamp_value)
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._locked_bases   = {
            name: basis.to(target_device)
            for name, basis in sd["locked_bases"].items()
        }
        print(
            f"[EL] Resumed. Basis {'captured' if self.basis_captured else 'not yet captured'} "
            f"at step {self.capture_step}. "
            f"{len(self._locked_bases)} matrices locked.",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPECTRAL SPARSITY ENFORCER (SSE)
# ─────────────────────────────────────────────────────────────────────────────

def _stable_rank(W: torch.Tensor) -> float:
    S = torch.linalg.svdvals(W.float())
    return (torch.norm(W, "fro") ** 2 / (S[0] ** 2 + 1e-8)).item()

def _sv_variance(W: torch.Tensor) -> float:
    S = torch.linalg.svdvals(W.float())
    return S.var().item()

def _spectral_reinflate(W: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    boost = eps * torch.exp(-S / (S.mean() + 1e-8))
    S_new = S + boost
    return (U * S_new.unsqueeze(0)) @ Vh

def _truncate_noise_tail(W: torch.Tensor, threshold_fraction: float = 0.01) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    mask  = (S > threshold_fraction * S[0]).float()
    S_clean = S * mask
    return (U * S_clean.unsqueeze(0)) @ Vh


class SpectralSparsityEnforcer:
    def __init__(
        self,
        all_params:  List[Tuple[str, nn.Parameter]],
        check_every: int = 50,
    ):
        self.all_params   = all_params
        self.check_every  = check_every
        self.step_count   = 0
        self._baseline_sr:  Dict[str, float] = {}
        self._baseline_var: Dict[str, float] = {}
        self._prev_norms:   Dict[str, float] = {}

    def capture_baselines(self, verbose: bool = True):
        for name, p in self.all_params:
            if p.dim() < 2:
                continue
            self._baseline_sr[name]  = _stable_rank(p.data)
            self._baseline_var[name] = _sv_variance(p.data)
            self._prev_norms[name]   = torch.norm(p.data, "fro").item()
        if verbose:
            print(f"[SSE] Baselines captured for {len(self._baseline_sr)} matrices.", flush=True)

    def step_check(self, phase: int, verbose: bool = True):
        self.step_count += 1
        if self.step_count % self.check_every != 0:
            return
        if phase < 2:
            return

        corrections = []
        for name, p in self.all_params:
            if p.dim() < 2 or name not in self._baseline_sr:
                continue

            sr     = _stable_rank(p.data)
            var    = _sv_variance(p.data)
            fn     = torch.norm(p.data, "fro").item()
            growth = (fn - self._prev_norms.get(name, fn)) / (self._prev_norms.get(name, fn) + 1e-8)
            self._prev_norms[name] = fn

            if sr < 0.80 * self._baseline_sr[name]:
                if phase == 4:
                    W_new = _spectral_reinflate(p.data, eps=1e-3)
                    p.data.copy_(W_new.to(p.dtype))
                    corrections.append(f"{name}:rank_collapse_corrected")
                else:
                    corrections.append(f"{name}:rank_collapse_DETECTED")

            if var < 0.10 * self._baseline_var.get(name, var + 1e-8):
                corrections.append(f"{name}:energy_equal(increase_entmax_alpha)")

            if growth > 0.05:
                if phase == 4:
                    W_new = _truncate_noise_tail(p.data, threshold_fraction=0.01)
                    p.data.copy_(W_new.to(p.dtype))
                    corrections.append(f"{name}:rank_creep_truncated")
                else:
                    corrections.append(f"{name}:rank_creep_DETECTED")

        if corrections and verbose:
            print(f"[SSE step {self.step_count}] " + " | ".join(corrections), flush=True)

    def state_dict(self) -> dict:
        return {
            "step_count":    self.step_count,
            "baseline_sr":   self._baseline_sr,
            "baseline_var":  self._baseline_var,
            "prev_norms":    self._prev_norms,
        }

    def load_state_dict(self, sd: dict):
        self.step_count    = sd["step_count"]
        self._baseline_sr  = sd["baseline_sr"]
        self._baseline_var = sd["baseline_var"]
        self._prev_norms   = sd["prev_norms"]
        print(f"[SSE] Resumed at step {self.step_count}.", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_thermodynamic_state(
    path:         str,
    accelerator:  TransitionAccelerator,
    eq_lock:      EquilibriumLock,
    sse:          SpectralSparsityEnforcer,
    muon:         Optional[MuonOptimizer] = None,
    extra:        Optional[dict] = None,
):
    state = {
        "accelerator": accelerator.state_dict(),
        "eq_lock":     eq_lock.state_dict(),
        "sse":         sse.state_dict(),
    }
    if muon is not None:
        state["muon"] = muon.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"[THERMO] State saved to {path}", flush=True)


def load_thermodynamic_state(
    path:        str,
    accelerator: TransitionAccelerator,
    eq_lock:     EquilibriumLock,
    sse:         SpectralSparsityEnforcer,
    muon:        Optional[MuonOptimizer] = None,
    device:      torch.device = None,
) -> dict:
    state = torch.load(path, map_location="cpu")
    thermo = state.get("thermo", state)

    accelerator.load_state_dict(thermo["accelerator"])
    eq_lock.load_state_dict(thermo["eq_lock"], device=device)
    sse.load_state_dict(thermo["sse"])
    if muon is not None and "muon" in thermo:
        muon.load_state_dict(thermo["muon"])

    print(f"[THERMO] State loaded from {path}", flush=True)
    return thermo
