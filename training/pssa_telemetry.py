import torch
import os
import warnings

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. PSSATelemetry will fallback to stdout/local logging.")

class PSSATelemetry:
    """
    Stage 35.5: Massive Instrumentation & Densification Monitoring.
    Tracks the internal health of the PSSA cognitive society.
    """
    def __init__(self, project_name="pssa_scaling", run_name=None, config=None, use_wandb=True, csv_path="logs/scaling_laws.csv", start_step=0):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.step_counter = start_step
        self.csv_path = csv_path
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Initialize CSV Header if new, or migrate existing header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("step,loss,ignition_rate,active_edges,pathway_entropy,cce,cosine_drift,trust_divergence,paradigm_pressure,introspection_fatigue,cognitive_pressure,decompose_triggered\n")
        else:
            with open(self.csv_path, 'r') as f:
                header = f.readline().strip()
            if "cognitive_pressure" not in header:
                with open(self.csv_path, 'r') as f:
                    lines = f.readlines()
                lines[0] = header + ",cognitive_pressure,decompose_triggered\n"
                with open(self.csv_path, 'w') as f:
                    f.writelines(lines)
        
        # Densification alarms
        self.ignition_warning_threshold = 0.15 # 15% densification is a severe risk
        self.k_star_active = 2.0
        self.frob_growth_rate = 0.0
        
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
            print(f"📡 PSSA Telemetry active: Logging to wandb project '{project_name}'")
        else:
            print(f"📡 PSSA Telemetry active: Local stdout/CSV logging mode.")

    def log_step(self, loss, model_outputs, sse=None, eq_lock=None):
        """
        Extracts and logs the deep cognitive metrics from the PSSAGPT forward pass.
        model_outputs: The huge tuple returned by PSSAGPT.forward()
        """
        (logits_tensor, out_gates, out_slots, out_pre_wave, out_adj, out_entities, out_retrievals, out_writes, out_scopes, recon_loss,
         out_trust, out_recency, out_ignition, out_fatigue, out_drift, out_trust_div, out_paradigm) = model_outputs
         
        self.step_counter += 1
        
        # Calculate cognitive metrics across the batch for the final step of the sequence
        # We assume out_ignition is a list of tensors per layer: [batch, seq_len] or similar
        # Actually, PSSAGPT currently returns lists of states per layer over the sequence.
        # Let's extract the very last layer, very last timestep metrics.
        
        layer_idx = -1 # Final layer
        
        # 1. Sparse Routing Health (Hidden Densification Monitoring)
        # out_ignition[layer][-1] -> shape [batch]
        last_ignitions = out_ignition[layer_idx][-1].float()
        ignition_rate = last_ignitions.mean().item()
        
        # Active edges in the graph attention
        # out_adj is [batch, seq, num_slots, num_slots]
        last_adj = out_adj[layer_idx][-1]
        active_edges = (last_adj > 0.05).float().sum(dim=-1).mean().item()
        
        # STAGE 35.5 (PHASE 4): Sparse Camouflage Detection (Pathway Entropy)
        # We track the entropy of the attention distribution across the sequence.
        # If it collapses to 0, the architecture is locked in a static corridor (Fake Sparsity)
        adj_seq = out_adj[layer_idx] # [batch, seq, num_slots, num_slots]
        mean_adj_seq = adj_seq.mean(dim=(0, 2)) # [seq, num_slots]
        mean_adj_seq = mean_adj_seq / (mean_adj_seq.sum(dim=-1, keepdim=True) + 1e-9)
        pathway_entropy = - (mean_adj_seq * torch.log(mean_adj_seq + 1e-9)).sum(dim=-1).mean().item()
        
        # Calculate scope weights entropy across layers
        # out_scopes has shape [batch, seq_len, num_layers, num_scopes]
        last_scopes = out_scopes[:, -1, :, :] # [batch, num_layers, num_scopes]
        scope_ent_by_layer = - (last_scopes * torch.log(last_scopes + 1e-9)).sum(dim=-1) # [batch, num_layers]
        scope_entropy = scope_ent_by_layer.mean().item()
        
        # 2. Epistemic Health (Drift & Poisoning Monitoring)
        # out_drift[layer][-1] -> [batch, 1]
        mean_drift = out_drift[layer_idx][-1].mean().item()
        mean_trust_div = out_trust_div[layer_idx][-1].mean().item()
        mean_paradigm_pressure = out_paradigm[layer_idx][-1].mean().item()
        
        # STAGE 35.5 (PHASE 4): Cognitive Thermodynamics (CCE)
        # CCE = Useful State Changes / Total Synchronization Events
        # We use out_entities magnitude change as Useful State Change
        E_t = out_entities[layer_idx][-1] # [batch, scopes, entities, d_model]
        E_t_prev = out_entities[layer_idx][-2] if out_entities[layer_idx].size(0) > 1 else E_t
        delta_E = (E_t - E_t_prev).pow(2).mean().item()
        cce = delta_E / (active_edges + 1e-9)
        
        # 3. Society Health
        # out_fatigue[layer][-1] -> [batch, 1]
        mean_fatigue = out_fatigue[layer_idx][-1].mean().item()
        
        # Extract spectral metrics from SSE / EquilibriumLock if available (only every 100 steps to prevent training bottleneck)
        if self.step_counter % 100 == 0 or self.step_counter <= 1:
            active_modes = []
            growth_rates = []
            
            if sse is not None:
                for name, p in sse.all_params:
                    if p.dim() < 2 or name not in sse._baseline_sr:
                        continue
                    S = torch.linalg.svdvals(p.data.float())
                    active = (S > 0.01 * S[0]).sum().item()
                    active_modes.append(active)
                    
                    fn = torch.norm(p.data, "fro").item()
                    prev_fn = sse._prev_norms.get(name, fn)
                    growth = (fn - prev_fn) / (prev_fn + 1e-8)
                    growth_rates.append(growth)
            elif eq_lock is not None and eq_lock.basis_captured:
                all_params = eq_lock.routing_params + eq_lock.worldmodel_params
                for name, p in all_params:
                    if name not in eq_lock._k_star or p.dim() < 2:
                        continue
                    S = torch.linalg.svdvals(p.data.float())
                    active = (S > 0.01 * S[0]).sum().item()
                    active_modes.append(active)
                    
            if active_modes:
                self.k_star_active = sum(active_modes) / len(active_modes)
            if growth_rates:
                self.frob_growth_rate = sum(growth_rates) / len(growth_rates)
        
        # Compute cognitive pressure
        P, decompose_triggered = self.cognitive_pressure(
            ignition_rate=ignition_rate,
            scope_entropy=scope_entropy,
            k_star_active=self.k_star_active,
            frob_growth_rate=self.frob_growth_rate
        )
        
        metrics = {
            "train/loss": loss,
            "sparse_health/ignition_rate": ignition_rate,
            "sparse_health/active_edges": active_edges,
            "sparse_health/pathway_entropy": pathway_entropy,
            "thermodynamics/cce": cce,
            "epistemic_health/cosine_drift": mean_drift,
            "epistemic_health/trust_divergence": mean_trust_div,
            "epistemic_health/paradigm_pressure": mean_paradigm_pressure,
            "society_health/introspection_fatigue": mean_fatigue,
            "cognitive_pressure/pressure": P,
            "cognitive_pressure/decompose_triggered": int(decompose_triggered)
        }
        
        # Append to CSV for persistent offline analysis
        with open(self.csv_path, 'a') as f:
            f.write(f"{self.step_counter},{loss:.4f},{ignition_rate:.4f},{active_edges:.4f},{pathway_entropy:.4f},{cce:.4f},{mean_drift:.4f},{mean_trust_div:.4f},{mean_paradigm_pressure:.4f},{mean_fatigue:.4f},{P:.4f},{int(decompose_triggered)}\n")
        
        if self.use_wandb:
            wandb.log(metrics, step=self.step_counter)
            
        # Densification Alarms
        if ignition_rate > self.ignition_warning_threshold:
            warnings.warn(f"🚨 DENSIFICATION ALARM: Ignition rate climbed to {ignition_rate*100:.2f}%. "
                          f"The sparse cognitive society is collapsing into a dense transformer.")
            
        if self.step_counter % 100 == 0:
            print(f"[PRESSURE] P={P:.3f} {'⚠️ DECOMPOSE' if decompose_triggered else 'OK'}", flush=True)
            if not self.use_wandb:
                print(f"Step {self.step_counter} | Loss: {loss:.4f} | Ign: {ignition_rate*100:.1f}% | "
                      f"Drift: {mean_drift:.3f} | Pressure: {mean_paradigm_pressure:.3f}")

    def cognitive_pressure(self, ignition_rate, scope_entropy, k_star_active, frob_growth_rate,
                           k_star_target=2, tau=0.35):
        # Normalize each signal to [0, 1]
        p_ignition = min(ignition_rate / 0.33, 1.0)    # 33% = full alarm
        p_entropy  = 1.0 - min(scope_entropy / 2.0, 1.0) # low entropy = bad
        p_kstar    = min(max(k_star_active - k_star_target, 0) / k_star_target, 1.0)
        p_frob     = min(frob_growth_rate / 0.10, 1.0)  # 10%/epoch = high creep

        # Weighted combination — ignition and entropy are most diagnostic
        P = 0.35 * p_ignition + 0.30 * p_entropy + 0.20 * p_kstar + 0.15 * p_frob

        return P, P > tau  # scalar pressure + boolean trigger

    def finish(self):
        if self.use_wandb:
            wandb.finish()
