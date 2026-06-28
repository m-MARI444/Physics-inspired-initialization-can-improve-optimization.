import os
import sys
# Must be set BEFORE torch is imported so the CUDA caching allocator uses
# expandable segments, which eliminates the OOM caused by fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Prevent HuggingFace datasets from caching shards in RAM
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# Use truststore to access OS certificates and fallback to unverified contexts if needed
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Disable CUDA Graph recording for dynamic shapes and assume static shapes by default
# to prevent cudagraph partitioning and compilation overhead.
try:
    import torch._inductor.config
    import torch._dynamo.config
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._dynamo.config.assume_static_by_default = True
except Exception:
    pass
try:
    import bitsandbytes as bnb
    AdamW8bit = bnb.optim.AdamW8bit
    if os.environ.get("RANK", "0") == "0":
        print("[OPT] bitsandbytes available. Will use 8-bit AdamW to save ~400MB VRAM.")
except ImportError:
    AdamW8bit = AdamW
    if os.environ.get("RANK", "0") == "0":
        print("[OPT] bitsandbytes not found. Using standard AdamW.")
import torch.amp as amp
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# Global monkeypatch to disable SSL verification for requests and httpx (works on all versions of huggingface_hub/datasets)
import requests
import urllib3
import httpx
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# requests patch
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# httpx Client patch
original_httpx_init = httpx.Client.__init__
def patched_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_httpx_init

# httpx AsyncClient patch
original_httpx_async_init = httpx.AsyncClient.__init__
def patched_httpx_async_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = patched_httpx_async_init
import time
import math
import signal
import gc
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT
from training.pssa_telemetry import PSSATelemetry

class PSSAGatherDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        if not outputs:
            return outputs

        # outputs is a list of 17-element tuples, one per GPU replica.
        # PSSAGPT.forward() returns:
        # (logits_tensor[0], out_gates[1], out_slots[2], out_pre_wave[3], out_adj[4],
        #  out_entities[5], out_retrievals[6], out_writes[7], out_scopes[8], recon_loss[9],
        #  out_trust[10], out_recency[11], out_ignition[12], out_fatigue[13],
        #  out_drift[14], out_trust_div[15], out_paradigm[16])

        # 1. Concatenate logits across the batch dimension
        logits_gathered = torch.cat([out[0].to(output_device) for out in outputs], dim=0)

        # 2. Concatenate batch-dimensioned tensors: out_gates, out_slots, out_pre_wave, out_adj,
        #    out_entities, out_retrievals, out_writes, out_scopes
        batch_tensors = {}
        for idx in [1, 2, 3, 4, 5, 6, 7, 8]:
            batch_tensors[idx] = torch.cat([out[idx].to(output_device) for out in outputs], dim=0)

        # 3. Average the reconstruction loss scalar across replicas
        recon_gathered = torch.stack([out[9].to(output_device).view(-1).mean() for out in outputs]).mean()

        # 4. Concatenate telemetry tensors (trust, recency, ignition, fatigue, drift, trust_div, paradigm)
        #    These are [batch, seq_len, num_layers, ...] tensors — concat on batch dim
        telemetry_tensors = {}
        for idx in [10, 11, 12, 13, 14, 15, 16]:
            try:
                telemetry_tensors[idx] = torch.cat([out[idx].to(output_device) for out in outputs], dim=0)
            except Exception:
                # Fall back to replica 0 if shape is unexpected
                telemetry_tensors[idx] = outputs[0][idx]

        # Assemble gathered output from replica 0 as base, then overwrite specific indices
        gathered = list(outputs[0])
        gathered[0] = logits_gathered
        for idx in [1, 2, 3, 4, 5, 6, 7, 8]:
            gathered[idx] = batch_tensors[idx]
        gathered[9] = recon_gathered
        for idx in [10, 11, 12, 13, 14, 15, 16]:
            gathered[idx] = telemetry_tensors[idx]

        return tuple(gathered)

from training.thermodynamics import (
    MuonOptimizer,
    TransitionAccelerator,
    EquilibriumLock,
    SpectralSparsityEnforcer,
    save_thermodynamic_state,
    load_thermodynamic_state,
)

# --- Global Variables for Pause and Play ---
is_paused = False

def signal_handler(sig, frame):
    global is_paused
    print("\n🛑 Pause signal received! Finishing current step and saving checkpoint safely...")
    is_paused = True

def separate_parameter_groups(model):
    ROUTING_KEYWORDS    = ["gate_proj", "entity_route", "scope_net", "isolate_gate"]
    WORLDMODEL_KEYWORDS = ["update_net", "pred_net", "out_proj"]

    routing_params    = []
    worldmodel_params = []
    other_params      = []

    # Unwrap DataParallel so named_parameters() returns clean names without 'module.' prefix
    raw_model = model.module if hasattr(model, 'module') else model

    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        is_routing    = any(k in name for k in ROUTING_KEYWORDS)
        is_worldmodel = any(k in name for k in WORLDMODEL_KEYWORDS)

        if is_routing:
            routing_params.append((name, p))
        elif is_worldmodel:
            worldmodel_params.append((name, p))
        else:
            other_params.append((name, p))

    import os
    if os.environ.get("RANK", "0") == "0":
        print(f"[PARAM GROUPS] routing={len(routing_params)} "
              f"worldmodel={len(worldmodel_params)} "
              f"other={len(other_params)}", flush=True)

    return routing_params, worldmodel_params, other_params

def build_optimizers(model, phase=1, lr=3e-4, weight_decay=1e-2, backend="ddp"):
    raw_model = model.module if hasattr(model, 'module') else model
    import os

    if backend != "ddp":
        # FSDP / DeepSpeed do not support split Muon optimizer in Phase 3.
        # We use a single unified AdamW optimizer across all phases.
        import torch.optim as optim
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if os.environ.get("RANK", "0") == "0":
            print(f"[OPT] Backend {backend}: Unified AdamW on all params (all phases)", flush=True)
        return optimizer, None

    routing_params, worldmodel_params, other_params = separate_parameter_groups(model)
    if phase < 3:
        # Preserve original parameter order to match the checkpoint's state dict mappings
        # Use 8-bit AdamW to save ~400MB VRAM on the optimizer state buffers
        optimizer  = AdamW8bit(
            raw_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if os.environ.get("RANK", "0") == "0":
            print(f"[OPT] Phase {phase}: AdamW on all params (pre-grokking)", flush=True)
        return optimizer, None
    else:
        muon_opt = MuonOptimizer(
            routing_params    = routing_params,
            other_params      = worldmodel_params + other_params,
            lr_muon           = 0.02,
            lr_adam           = lr,
            weight_decay      = weight_decay,
        )
        if os.environ.get("RANK", "0") == "0":
            print(f"[OPT] Phase {phase}: Muon(routing k*=1) + AdamW(world model k*=2)", flush=True)
        return None, muon_opt

def build_thermodynamic_controllers(model, device):
    routing_params, worldmodel_params, _ = separate_parameter_groups(model)
    sentinel_params = routing_params

    accelerator = TransitionAccelerator(
        sentinel_params = sentinel_params,
        gap_threshold   = 0.35,
        window          = 20,
        check_every     = 100,
        wd_boost_factor = 3.0,
    )

    eq_lock = EquilibriumLock(
        routing_params    = routing_params,
        worldmodel_params = worldmodel_params,
        lock_strength     = 0.1,
        clamp_value       = 10.0,
    )

    sse = SpectralSparsityEnforcer(
        all_params  = routing_params + worldmodel_params,
        check_every = 50,
    )

    return accelerator, eq_lock, sse

def save_campaign_checkpoint(
    path, model, optimizer, muon_opt,
    accelerator, eq_lock, sse,
    step, dataset_items_consumed, loss_val, phase,
    backend="ddp", hf_repo=None, hf_token=None
):
    import gc
    import os
    # Force GC and empty CUDA cache beforehand to clear any leftover memory
    torch.cuda.empty_cache()
    gc.collect()

    is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        is_fsdp = isinstance(model, FSDP)
    except ImportError:
        pass

    # Move model state dict to CPU explicitly, tensor by tensor
    if is_fsdp:
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
    else:
        model_state = {}
        raw_model = model.module if hasattr(model, 'module') else model
        for k, v in raw_model.state_dict().items():
            model_state[k] = v.cpu()

    thermo_state = {
        'accelerator': accelerator.state_dict(),
        'eq_lock':     eq_lock.state_dict(),
        'sse':         sse.state_dict(),
    }

    checkpoint = {
        'global_step': step,
        'dataset_items_consumed': dataset_items_consumed,
        'loss': loss_val,
        'phase': phase,
        'model_state_dict': model_state,
        'thermo': thermo_state
    }

    if optimizer is not None:
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            cpu_opt_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
            checkpoint['optimizer_state_dict'] = cpu_opt_state
        else:
            opt_state = optimizer.state_dict()
            cpu_opt_state = {'state': {}, 'param_groups': opt_state['param_groups']}
            for p, s in opt_state['state'].items():
                cpu_opt_state['state'][p] = {
                    sk: (sv.cpu() if isinstance(sv, torch.Tensor) else sv)
                    for sk, sv in s.items()
                }
            checkpoint['optimizer_state_dict'] = cpu_opt_state

    if muon_opt is not None:
        # MuonOptimizer.state_dict() already moves its internal states to CPU
        checkpoint['thermo']['muon'] = muon_opt.state_dict()

    global_rank = int(os.environ.get("RANK", "0"))
    if global_rank == 0:
        # Save serialization stream to file atomically
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"💾 Saved checkpoint at step {step}, phase={phase}", flush=True)

        if hf_repo and hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                print(f"📡 Uploading checkpoint to Hugging Face repository '{hf_repo}'...", flush=True)
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo="pssa_campaign_latest.pth",
                    repo_id=hf_repo,
                    repo_type="model",
                    token=hf_token
                )
                print("✅ Hugging Face upload complete!", flush=True)
            except Exception as e:
                print(f"[HF WARNING] Failed to upload checkpoint to Hugging Face Hub: {e}", flush=True)
    
    # Deallocate large memory structures immediately
    del checkpoint, model_state, thermo_state
    if optimizer is not None:
        del cpu_opt_state
    torch.cuda.empty_cache()
    gc.collect()

def load_campaign_checkpoint(
    path, model, device,
    accelerator, eq_lock, sse,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    thermo = ckpt.get("thermo", {})
    if "accelerator" in thermo:
        accelerator.load_state_dict(thermo["accelerator"])
    if "eq_lock" in thermo:
        eq_lock.load_state_dict(thermo["eq_lock"], device=device)
    if "sse" in thermo:
        sse.load_state_dict(thermo["sse"])

    phase = ckpt.get("phase", 1)
    import os
    if os.environ.get("RANK", "0") == "0":
        print(f"▶️ Loaded step={ckpt['global_step']} phase={phase}", flush=True)
    return ckpt['global_step'], ckpt.get('dataset_items_consumed', 0), ckpt.get('loss', 0.0), phase, ckpt

def run_relational_distillation(model, teacher_name, get_batch_fn, device, hf_token=None, num_steps=250, global_rank=0):
    if global_rank == 0:
        print(f"\n🧠 Starting PSSA Relational Distillation Phase (Teacher: {teacher_name}, Steps: {num_steps})...", flush=True)
    from transformers import GPT2LMHeadModel
    
    # 1. Load teacher model
    try:
        teacher = GPT2LMHeadModel.from_pretrained(
            teacher_name, 
            token=hf_token, 
            attn_implementation="eager"
        ).to(device)
        teacher.eval()
    except Exception as e:
        if global_rank == 0:
            print(f"Error loading teacher model {teacher_name}: {e}", flush=True)
        return
        
    # 2. Setup distillation optimizer (only optimizing the student model)
    raw_model = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 3. Import energy modules
    from pssa.energy import compute_relational_energy
    
    pbar = range(num_steps)
    if global_rank == 0:
        try:
            from tqdm import tqdm
            pbar = tqdm(pbar, desc="Distillation Warmup")
        except ImportError:
            pass
            
    raw_model.train()
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Get next batch
        input_ids, targets = get_batch_fn()
        
        # Forward pass teacher
        with torch.no_grad():
            outputs_t = teacher(input_ids, output_attentions=True, output_hidden_states=True)
            t_logits = outputs_t.logits
            t_attns = outputs_t.attentions # Tuple of attention maps [batch, num_heads, T, T]
            
        # Forward pass student (run eager mode for rapid step latency)
        s_outputs = raw_model(input_ids, return_telemetry=True)
        s_logits = s_outputs[0]
        s_retrievals = s_outputs[6] # [batch, T, M]
        recon_loss = s_outputs[9]
        
        # Compute logit alignment (prediction energy)
        loss_pred = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), targets.view(-1))
        
        # Compute relational distillation energy across layers
        loss_relation = torch.tensor(0.0, device=device)
        layers_count = 0
        
        if step == 0 and global_rank == 0:
            print(f"\n[DEBUG DISTILL] teacher_attentions count: {len(t_attns)}, layer 0 attention shape: {t_attns[0].shape}")
            print(f"[DEBUG DISTILL] student_retrievals shape: {s_retrievals.shape}")
            
        for l, layer in enumerate(raw_model.layers):
            dep_matrix = getattr(layer, "last_dependency", None)
            if step == 0 and global_rank == 0:
                print(f"[DEBUG DISTILL] Layer {l} last_dependency type/shape: {type(dep_matrix)} / {dep_matrix.shape if dep_matrix is not None else 'None'}")
            
            if dep_matrix is not None:
                t_layer_idx = min(2 * l + 1, len(t_attns) - 1)
                A_t = t_attns[t_layer_idx] # [batch, num_heads, T, T]
                
                loss_relation = loss_relation + compute_relational_energy(dep_matrix, A_t, s_retrievals)
                layers_count += 1
                
        if layers_count > 0:
            loss_relation = loss_relation / layers_count
            
        # Total distillation loss (Connection L2 regularization is handled natively by AdamW weight_decay)
        total_loss = loss_pred + 1.0 * loss_relation + 0.1 * recon_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        
        if global_rank == 0 and step % 50 == 0:
            print(f"  [Distill Step {step:3d}] Pred={loss_pred.item():.4f} | Rel={loss_relation.item():.4f}", flush=True)
            
    if global_rank == 0:
        print("✨ PSSA Relational Distillation initialization completed successfully!\n", flush=True)
        
    # Free teacher and clean up memory
    del teacher, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def run_campaign():
    global is_paused
    torch.set_float32_matmul_precision('high')
    
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Parse CLI arguments early
    import sys
    batch_size = 16
    disable_compile = False
    dataset_type = "fineweb"
    backend = "ddp"
    hf_repo = None
    hf_token = None
    d_model = 192
    num_layers = 6
    num_slots = 8
    max_steps = 50000
    distill_teacher = None
    distill_steps = 250
    for arg in sys.argv:
        if arg.startswith("--batch_size="):
            batch_size = int(arg.split("=")[1])
        elif arg == "--disable_compile":
            disable_compile = True
        elif arg.startswith("--dataset="):
            dataset_type = arg.split("=")[1].lower()
        elif arg.startswith("--backend="):
            backend = arg.split("=")[1].lower()
        elif arg.startswith("--hf_repo="):
            hf_repo = arg.split("=")[1].strip()
        elif arg.startswith("--hf_token="):
            hf_token = arg.split("=")[1].strip()
        elif arg.startswith("--d_model="):
            d_model = int(arg.split("=")[1])
        elif arg.startswith("--num_layers="):
            num_layers = int(arg.split("=")[1])
        elif arg.startswith("--num_slots="):
            num_slots = int(arg.split("=")[1])
        elif arg.startswith("--max_steps="):
            max_steps = int(arg.split("=")[1])
        elif arg.startswith("--distill_teacher="):
            distill_teacher = arg.split("=")[1].strip()
        elif arg.startswith("--distill_steps="):
            distill_steps = int(arg.split("=")[1])

    if global_rank == 0:
        print(f"🚀 Starting Scientific Mapping Campaign on {device} (Backend={backend}, DDP={is_ddp})")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # ── Step 1: Pre-check checkpoint for items consumed ──────────────────────────
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = os.path.join(checkpoint_dir, "pssa_campaign_latest.pth")

    # If the local checkpoint file is missing but HF credentials are provided, attempt to download it on Rank 0 first
    if global_rank == 0 and not os.path.exists(latest_checkpoint) and hf_repo and hf_token:
        print(f"📡 Local checkpoint not found. Attempting to download from Hugging Face repository '{hf_repo}'...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            import inspect
            sig = inspect.signature(hf_hub_download)
            kwargs = {
                "repo_id": hf_repo,
                "filename": "pssa_campaign_latest.pth",
                "local_dir": checkpoint_dir,
            }
            if "token" in sig.parameters:
                kwargs["token"] = hf_token
            if "use_auth_token" in sig.parameters:
                kwargs["use_auth_token"] = hf_token
            
            hf_hub_download(**kwargs)
            print("✅ Downloaded latest checkpoint from Hugging Face Hub successfully.", flush=True)
        except Exception as e:
            import traceback
            print("[DEBUG] hf_hub_download kwargs:", {k: (v if k != "token" and k != "use_auth_token" else "CLASSIFIED") for k, v in kwargs.items()}, flush=True)
            traceback.print_exc()
            print(f"[HF INFO] Could not download checkpoint from Hugging Face Hub: {e}. Starting fresh or checking other paths.", flush=True)

    # Sync all ranks: wait until Rank 0 has finished downloading before checking file existence
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    items_to_skip = 0
    checkpoint_corrupted = False
    if os.path.exists(latest_checkpoint):
        try:
            pre_ckpt = torch.load(latest_checkpoint, map_location="cpu")
            items_to_skip = pre_ckpt.get("dataset_items_consumed", 0)
            del pre_ckpt
            gc.collect()
        except Exception as e:
            print(f"[CAMPAIGN WARNING] Checkpoint '{latest_checkpoint}' is corrupted or failed to load: {e}", flush=True)
            corrupted_backup = latest_checkpoint + ".corrupted"
            print(f"[CAMPAIGN] Renaming corrupted checkpoint to '{corrupted_backup}' and starting from scratch...", flush=True)
            try:
                os.replace(latest_checkpoint, corrupted_backup)
            except Exception:
                pass
            checkpoint_corrupted = True

    # ── Step 2: NOW load model and checkpoint onto GPU ─────────────────────────
    # (d_model, num_layers, and num_slots are parsed dynamically from sys.argv)
    
    if global_rank == 0:
        # Dynamically measure parameter count
        test_m = PSSAGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            num_slots=num_slots,
            tau=0.15,
            num_scopes=3,
            num_layers=num_layers
        )
        total_params = sum(p.numel() for p in test_m.parameters())
        print(f"Initializing PSSA Model with size: {total_params / 1e6:.2f} Million Parameters (d_model={d_model}, layers={num_layers}, slots={num_slots})...")
        del test_m
    model = PSSAGPT(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_slots=num_slots, 
        tau=0.15, 
        num_scopes=3, 
        num_layers=num_layers
    ).to(device)
    
    accelerator, eq_lock, sse = build_thermodynamic_controllers(model, device)
    
    global_step = 0
    dataset_items_consumed = 0
    phase = 1
    ckpt = None

    def wrap_model(m):
        if not is_ddp:
            return m
        if backend == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            return DDP(m, device_ids=[local_rank], find_unused_parameters=True)
        elif backend == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            import functools
            from pssa_project.model.pssa_gpt import PSSALayer
            
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={PSSALayer}
            )
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            return FSDP(
                m,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mixed_precision_policy,
                device_id=torch.cuda.current_device(),
                sync_module_states=True
            )
        return m
    
    if os.path.exists(latest_checkpoint) and not checkpoint_corrupted:
        try:
            global_step, dataset_items_consumed, _, phase, ckpt = load_campaign_checkpoint(
                latest_checkpoint, model, device, accelerator, eq_lock, sse
            )
            model = wrap_model(model)
            optimizer, muon_opt = build_optimizers(model, phase=phase, backend=backend)
            
            if optimizer is not None and ckpt is not None and 'optimizer_state_dict' in ckpt:
                if backend == "fsdp":
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    full_osd = ckpt['optimizer_state_dict']
                    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
                    optimizer.load_state_dict(sharded_osd)
                else:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            if muon_opt is not None and ckpt is not None and 'thermo' in ckpt and 'muon' in ckpt['thermo']:
                muon_opt.load_state_dict(ckpt['thermo']['muon'])
            if ckpt is not None:
                del ckpt
            torch.cuda.empty_cache()
            gc.collect()  # free CPU tensors from checkpoint load
            if global_rank == 0:
                print(f"Resumed at Step {global_step}")
        except Exception as e:
            if global_rank == 0:
                print(f"[CAMPAIGN WARNING] Failed to load checkpoint in full: {e}. Starting from scratch...", flush=True)
            global_step = 0
            dataset_items_consumed = 0
            phase = 1
            model = wrap_model(model)
            optimizer, muon_opt = build_optimizers(model, phase=1, backend=backend)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        if global_rank == 0:
            print("▶️ Starting fresh campaign from Step 0")
        model = wrap_model(model)
        optimizer, muon_opt = build_optimizers(model, phase=1, backend=backend)

    # Initialize DeepSpeed engine if selected
    if is_ddp and backend == "deepspeed":
        import deepspeed
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": True
            },
            "steps_per_print": 2000
        }
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=ds_config
        )

    import signal
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP) if hasattr(signal, "SIGHUP") else (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)
    
    telemetry = PSSATelemetry(use_wandb=False, start_step=global_step)
    model.train()
    scaler = amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    if is_ddp:
        if global_rank == 0:
            print(f"[CAMPAIGN] 🚀 Running in {backend.upper()} mode across {world_size} GPUs.", flush=True)
    elif torch.cuda.device_count() > 1:
        print(f"[CAMPAIGN] ℹ️ {torch.cuda.device_count()} GPUs detected. Running on primary GPU only by default. Run with torchrun to enable distributed backends.", flush=True)
    else:
        print("[CAMPAIGN] Running on a single GPU.", flush=True)

    if global_rank == 0:
        print(f"[CAMPAIGN] Configured Batch Size: {batch_size}", flush=True)

    # seq_len=64: optimal for single T4 GPU (15GB VRAM).
    seq_len = 64
    BUFFER_SIZE = 1000
    data_buffer = []
    buffer_ptr = 0

    # torch.compile is enabled by default now that L3 memory manager has zero CPU-GPU sync stalls!
    if not disable_compile and backend != "deepspeed":
        if global_rank == 0:
            print("[CAMPAIGN] ✅ torch.compile enabled (reduce-overhead mode) — expecting 2-4x speedup.", flush=True)
        model_fn = torch.compile(model, mode="reduce-overhead")
    else:
        if global_rank == 0:
            msg = "[CAMPAIGN] Running in eager mode (deepspeed backend)." if backend == "deepspeed" else "[CAMPAIGN] ℹ️ Running in eager mode (torch.compile disabled)."
            print(msg, flush=True)
        model_fn = model

    if global_rank == 0:
        print(f"[DATA] Selected dataset mode: {dataset_type}", flush=True)

    # Load dataset and skip to consumption point once at startup to prevent memory leaks from repeated skipping
    if global_rank == 0:
        print(f"[DATA] Initializing dataset stream at index {dataset_items_consumed}...", flush=True)
    if dataset_type == "hotpotqa":
        dataset_stream = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    else:
        dataset_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
    if is_ddp:
        from datasets.distributed import split_dataset_by_node
        dataset_stream = split_dataset_by_node(dataset_stream, rank=global_rank, world_size=world_size)
        
    skipped_ds = dataset_stream.skip(dataset_items_consumed)
    stream_iter = iter(skipped_ds)

    import queue
    import threading
    
    batch_queue = queue.Queue(maxsize=16)
    
    def data_loader_worker():
        nonlocal dataset_items_consumed, stream_iter
        local_consumed = dataset_items_consumed
        
        while True:
            # Fetch a buffer chunk of items
            new_buffer = []
            for _ in range(BUFFER_SIZE):
                try:
                    item = next(stream_iter)
                    local_consumed += 1
                    if dataset_type == "hotpotqa":
                        context_str = ""
                        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                            context_str += f"{title}: " + "".join(sentences) + "\n"
                        text = f"Context:\n{context_str}\nQuestion: {item['question']}\nAnswer: {item['answer']}"
                        new_buffer.append(text)
                    else:
                        new_buffer.append(item["text"])
                except StopIteration:
                    if dataset_type == "hotpotqa":
                        dataset_stream_fresh = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
                    else:
                        dataset_stream_fresh = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                    if is_ddp:
                        from datasets.distributed import split_dataset_by_node
                        dataset_stream_fresh = split_dataset_by_node(dataset_stream_fresh, rank=global_rank, world_size=world_size)
                    stream_iter = iter(dataset_stream_fresh)
                    try:
                        item = next(stream_iter)
                        local_consumed += 1
                        if dataset_type == "hotpotqa":
                            context_str = ""
                            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                                context_str += f"{title}: " + "".join(sentences) + "\n"
                            text = f"Context:\n{context_str}\nQuestion: {item['question']}\nAnswer: {item['answer']}"
                            new_buffer.append(text)
                        else:
                            new_buffer.append(item["text"])
                    except StopIteration:
                        break
            
            # Clean PyArrow/C++ allocations and run Python GC inside the loader thread
            gc.collect()
            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
                pa.jemalloc_memory_pool().release_unused()
                pa.system_memory_pool().release_unused()
            except Exception:
                pass
            
            # Tokenize in thread
            if new_buffer:
                batch_encodings = tokenizer(
                    [t[:1024] for t in new_buffer],
                    truncation=True,
                    max_length=seq_len + 1
                )
                local_buffer = [
                    ids for ids in batch_encodings["input_ids"]
                    if len(ids) >= seq_len + 1
                ]
            else:
                local_buffer = []
                
            # Put into queue in batches
            ptr = 0
            while ptr < len(local_buffer):
                batch_x_list = []
                batch_y_list = []
                # Accumulate batch_size items
                while len(batch_x_list) < batch_size and ptr < len(local_buffer):
                    tokens = local_buffer[ptr]
                    ptr += 1
                    if len(tokens) >= seq_len + 1:
                        batch_x_list.append(tokens[:seq_len])
                        batch_y_list.append(tokens[1:seq_len+1])
                
                if len(batch_x_list) == batch_size:
                    x_tensor = torch.tensor(batch_x_list, dtype=torch.long).pin_memory()
                    y_tensor = torch.tensor(batch_y_list, dtype=torch.long).pin_memory()
                    batch_queue.put((x_tensor, y_tensor, batch_size))

    # Start the background data loader thread
    loader_thread = threading.Thread(target=data_loader_worker, daemon=True)
    loader_thread.start()

    def get_next_batch():
        nonlocal dataset_items_consumed
        # Retrieve pre-loaded batch from queue (async, non-blocking GPU transfer)
        x_tensor, y_tensor, consumed_delta = batch_queue.get()
        dataset_items_consumed += consumed_delta
        return x_tensor.to(device, non_blocking=True), y_tensor.to(device, non_blocking=True)

    if not sse._baseline_sr:
        sse.capture_baselines(verbose=False)

    # ── Step-0 Relational Distillation Phase ───────────────────────
    if global_step == 0 and distill_teacher:
        run_relational_distillation(
            model=model,
            teacher_name=distill_teacher,
            get_batch_fn=get_next_batch,
            device=device,
            hf_token=hf_token,
            num_steps=distill_steps,
            global_rank=global_rank
        )

    # 50,000 steps total: from step 25,004 → only 25,000 more steps remain.
    # At ~3-5 sec/step after torch.compile: 25,000 × 4s = 100,000 sec ≈ 27 hours.
    # This fits within the 26-hour Kaggle free-session window.
    # The model at step 50K (Phase 3 GROKKING complete) is a fully trained, usable model.
    TARGET_STEPS = max_steps
    ema_dt = None
    
    loss_val = 0.0
    if global_rank == 0:
        print("\n--- Campaign Running (Press Ctrl+C to safely pause and save) ---", flush=True)
    while True:
        try:
            input_ids, targets = get_next_batch()
                
            start_t = time.time()
            
            if optimizer is not None:
                optimizer.zero_grad()
            if muon_opt is not None:
                muon_opt.zero_grad()
            
            log_telemetry = (global_step % 10 == 0)
            k_star_active = telemetry.k_star_active if hasattr(telemetry, 'k_star_active') else 2.0
            frob_growth_rate = telemetry.frob_growth_rate if hasattr(telemetry, 'frob_growth_rate') else 0.0
            
            k_star_active_t = torch.tensor(k_star_active, dtype=torch.float32, device=device)
            frob_growth_rate_t = torch.tensor(frob_growth_rate, dtype=torch.float32, device=device)
            
            locked_bases_dict = None
            lock_scale_val = 0.0
            if phase >= 3 and eq_lock.basis_captured:
                locked_bases_dict = eq_lock._locked_bases
                steps_since_capture = max(0, global_step - eq_lock.capture_step)
                lock_scale_val = min(1.0, float(steps_since_capture) / 100.0)
            lock_scale_t = torch.tensor(lock_scale_val, dtype=torch.float32, device=device)
            
            with amp.autocast('cuda' if device.type == "cuda" else 'cpu'):
                if log_telemetry:
                    # Run eager (uncompiled) model for the telemetry step to prevent compilation overhead
                    outputs = model(
                        input_ids, 
                        actor_mask=None, 
                        k_star_active=k_star_active_t, 
                        frob_growth_rate=frob_growth_rate_t,
                        return_telemetry=True,
                        locked_bases=locked_bases_dict,
                        lock_scale=lock_scale_t
                    )
                else:
                    # Run compiled model for maximum speed during normal training steps
                    outputs = model_fn(
                        input_ids, 
                        actor_mask=None, 
                        k_star_active=k_star_active_t, 
                        frob_growth_rate=frob_growth_rate_t,
                        return_telemetry=False,
                        locked_bases=locked_bases_dict,
                        lock_scale=lock_scale_t
                    )
                logits = outputs[0]
                recon_loss = outputs[9]
                
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                loss = ce_loss + recon_loss
                
            # lock_loss is now computed inside the forward pass to be DDP-compliant.
            # We track loss_lock value for logging and telemetry purposes.
            loss_lock = torch.tensor(0.0, device=device)
            if phase >= 3 and eq_lock.basis_captured and locked_bases_dict is not None:
                with torch.no_grad():
                    loss_lock = lock_scale_t * eq_lock.lock_loss()
                
            loss_val = loss.item()
            ce_loss_val = ce_loss.item()
            
            # Skip this step entirely if loss is non-finite — avoids gradient corruption.
            # NaN inputs trigger guards in the forward pass; we must not backward through them.
            if not math.isfinite(loss_val):
                print(f"[SKIP] Step {global_step}: non-finite loss={loss_val:.4f} — skipping update", flush=True)
                del loss, outputs, logits, recon_loss, ce_loss, input_ids, targets
                if optimizer is not None: optimizer.zero_grad()
                if muon_opt is not None: muon_opt.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1
                continue
            
            if global_rank == 0 and log_telemetry:
                telemetry.log_step(loss_val, outputs, sse=sse, eq_lock=eq_lock)
            del outputs, logits, recon_loss, ce_loss
            
            if backend == "deepspeed":
                model.backward(loss)
                model.step()
                
                prev_phase = phase
                phase = accelerator.step(base_wd=1e-2, verbose=(global_rank == 0 and global_step % 100 == 0))
                if phase == 3 and prev_phase < 3:
                    eq_lock.capture_basis(step=global_step)
                    if global_rank == 0:
                        print(f"[CAMPAIGN] Phase 3 activated at step {global_step}. Keeping DeepSpeed unified optimizer.", flush=True)
            else:
                if phase >= 3:
                    # Phase 3: plain fp32 backward — no scaler needed.
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                
                # Populate Adam v tracking for the Transition Accelerator
                if phase == 2 and optimizer is not None:
                    try:
                        for name, p in accelerator.sentinel_params:
                            if p.grad is not None and p in optimizer.state:
                                state = optimizer.state[p]
                                if 'exp_avg_sq' in state:
                                    accelerator.update_adam_v(name, state['exp_avg_sq'])
                    except Exception:
                        pass
                
                if optimizer is not None:
                    if phase < 3:
                        scaler.unscale_(optimizer)
                
                prev_phase = phase
                phase = accelerator.step(base_wd=1e-2, verbose=(global_rank == 0 and global_step % 100 == 0))
                
                if phase == 3 and prev_phase < 3:
                    eq_lock.capture_basis(step=global_step)
                    if backend == "ddp":
                        if optimizer is not None:
                            del optimizer
                            optimizer = None
                            torch.cuda.empty_cache()
                        optimizer, muon_opt = build_optimizers(model, phase=3, backend=backend)
                        torch.cuda.empty_cache()
                        if global_rank == 0:
                            print(f"[CAMPAIGN] Phase 3 activated at step {global_step}. Optimizer split.", flush=True)
                    else:
                        if global_rank == 0:
                            print(f"[CAMPAIGN] Phase 3 activated at step {global_step}. Keeping unified optimizer.", flush=True)
                    
                if optimizer is not None:
                    # Unwrap DataParallel to clip gradients on actual model parameters
                    _clip_params = (model.module if hasattr(model, 'module') else model).parameters()
                    torch.nn.utils.clip_grad_norm_(_clip_params, 1.0)
                    if phase < 3:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()  # plain step — no scaler in Phase 3
            
                if muon_opt is not None:
                    # Phase 3: backward was plain fp32, grads are already at correct scale.
                    # Check for gradient finiteness to prevent NaN propagation under sudden CE spikes
                    grads_finite = True
                    for name, p in muon_opt.routing_params + muon_opt.other_params:
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():
                                grads_finite = False
                                if global_rank == 0:
                                    print(f"[SKIP] Non-finite gradients detected in parameter {name} at step {global_step}! Skipping update.", flush=True)
                                break
                    
                    if grads_finite:
                        # 100-step linear warm-up for the new Phase 3 optimizers (Muon + AdamW) to mitigate Momentum Shock
                        steps_since_capture = max(0, global_step - eq_lock.capture_step)
                        if steps_since_capture < 100:
                            warmup_factor = min(1.0, float(steps_since_capture + 1) / 100.0)
                            muon_opt.lr_muon = 0.02 * warmup_factor
                            muon_opt.lr_adam = 3e-4 * warmup_factor
                        else:
                            muon_opt.lr_muon = 0.02
                            muon_opt.lr_adam = 3e-4

                        torch.nn.utils.clip_grad_norm_(
                            [p for _, p in muon_opt.routing_params + muon_opt.other_params], 1.0
                        )
                        muon_opt.step()
                    else:
                        muon_opt.zero_grad()
                
            sse.step_check(phase=phase, verbose=False)
            
            del loss, input_ids, targets
            # (torch.cuda.empty_cache() removed to prevent GPU stalls)
            # Periodic Python GC and PyArrow memory release to prevent heap fragmentation
            if global_step % 1000 == 0:
                gc.collect()
                try:
                    import pyarrow as pa
                    pa.jemalloc_memory_pool().release_unused()
                    pa.system_memory_pool().release_unused()
                except Exception:
                    pass
                try:
                    import ctypes
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except Exception:
                    pass
            elif global_step % 500 == 0:
                gc.collect()
            
            # Disable slow GC object tracking diagnostics in production to maximize throughput
            if False:
                import collections
                gc.collect()
                objs = gc.get_objects()
                counter = collections.Counter(type(o).__name__ for o in objs)
                print(f"[GC DIAG] Top objects: {counter.most_common(10)}", flush=True)
                tensors = [o for o in objs if isinstance(o, torch.Tensor)]
                print(f"[GC DIAG] Total tensors: {len(tensors)}", flush=True)
                if tensors:
                    sizes = collections.Counter(tuple(t.shape) for t in tensors)
                    print(f"[GC DIAG] Top tensor shapes: {sizes.most_common(5)}", flush=True)
                    target_tensors = [t for t in tensors if t.shape == (7885872,)]
                    print(f"[GC DIAG] Found {len(target_tensors)} tensors of shape (7885872,)", flush=True)
                    if hasattr(accelerator, '_grad_buffer'):
                        buffer_set = set(id(t) for t in accelerator._grad_buffer)
                        popped_tensors = [t for t in target_tensors if id(t) not in buffer_set]
                        print(f"[GC DIAG] Found {len(popped_tensors)} popped/unreferenced target tensors in GC", flush=True)
                        if len(popped_tensors) > 0:
                            ref_list = gc.get_referrers(popped_tensors[0])
                            print(f"[GC DIAG] Referrers of first popped target tensor (count {len(ref_list)}):", flush=True)
                            for r in ref_list:
                                if isinstance(r, dict):
                                    print(f"  - dict keys: {list(r.keys())[:10]}", flush=True)
                                elif isinstance(r, list):
                                    print(f"  - list length: {len(r)}", flush=True)
                                else:
                                    print(f"  - {type(r).__name__}: {str(r)[:100]}", flush=True)
                if hasattr(accelerator, '_grad_buffer'):
                    print(f"[GC DIAG] Accelerator buffer length: {len(accelerator._grad_buffer)} | maxlen: {accelerator._grad_buffer.maxlen}", flush=True)
                # Delete all diagnostic references so they don't persist in the loop frame
                try:
                    del objs, counter, tensors, sizes, target_tensors, popped_tensors
                except NameError:
                    pass
                try:
                    del ref_list
                except NameError:
                    pass
            
            dt = time.time() - start_t
            if ema_dt is None:
                ema_dt = dt
            else:
                ema_dt = 0.9 * ema_dt + 0.1 * dt
                
            if global_rank == 0 and global_step % 5 == 0:
                steps_remaining = TARGET_STEPS - global_step
                eta_seconds = steps_remaining * ema_dt
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"Step {global_step}/{TARGET_STEPS} | phase={phase} | CE={ce_loss_val:.4f} | Lock={loss_lock.item():.4f} | Total={loss_val:.4f} | RAM={ram_mb:.1f}MB | ETA: {eta_hours}h {eta_minutes}m")
                
            if global_rank == 0 and phase == 4 and global_step % 5000 == 0:
                healthy = eq_lock.spectral_health_check(verbose=True)
                if not healthy:
                    print(f"[CAMPAIGN WARNING] Re-densification detected at step {global_step}!", flush=True)
                    
            global_step += 1
            
            if global_step % 25 == 0 or is_paused:
                if global_rank == 0:
                    save_campaign_checkpoint(
                        latest_checkpoint, model, optimizer, muon_opt,
                        accelerator, eq_lock, sse,
                        global_step, dataset_items_consumed, loss_val, phase,
                        backend=backend, hf_repo=hf_repo, hf_token=hf_token
                    )
                if is_ddp:
                    import torch.distributed as dist
                    dist.barrier()
                if is_paused:
                    if global_rank == 0:
                        print("⏸️ Campaign paused successfully. You can resume later by running this script again.")
                    break
        except KeyboardInterrupt:
            if global_rank == 0:
                print("\n🛑 KeyboardInterrupt caught! Saving checkpoint safely before exit...", flush=True)
                save_campaign_checkpoint(
                    latest_checkpoint, model, optimizer, muon_opt,
                    accelerator, eq_lock, sse,
                    global_step, dataset_items_consumed, loss_val, phase,
                    backend=backend, hf_repo=hf_repo, hf_token=hf_token
                )
                print("⏸️ Campaign paused successfully. You can resume later by running this script again.", flush=True)
            if is_ddp:
                import torch.distributed as dist
                dist.barrier()
            break

if __name__ == "__main__":
    run_campaign()
