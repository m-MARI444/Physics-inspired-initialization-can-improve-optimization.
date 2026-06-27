import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# ==========================================
# 1. PSSA Language Model Architecture
# ==========================================

class PSSALanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_slots=5):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 1. Prediction Network (Predicts next token embedding from current slots)
        self.pred_net = nn.Sequential(
            nn.Linear(num_slots * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 2. Prediction-Error Gate
        self.gate_proj = nn.Linear(d_model, num_slots)
        
        # Base semantic inertia for the 5 hierarchical slots: 
        # Syntax, Entity, Dialogue, Semantic, Attention Focus
        self.register_buffer('base_inertia', torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]))
        self.inertia_momentum = 0.9
        self.inertia_scale = 1.0
        
        # 3. Slot Update Network
        self.update_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 4. Dynamic Relational Wave Propagation (Cross-Attention between slots)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wave_norm = nn.LayerNorm(d_model)
        
        # 5. Output Projection
        self.out_proj = nn.Linear(num_slots * d_model, vocab_size)
        
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x_emb = self.embed(input_ids) # [batch, seq_len, d_model]
        
        # Initialize Persistent Slots and Velocity (Inertia)
        S_t = torch.zeros(batch, self.num_slots, self.d_model, device=device)
        V_t = torch.zeros(batch, self.num_slots, device=device)
        
        gates = []
        logits = []
        slots = []
        
        for t in range(seq_len):
            x_t = x_emb[:, t, :] # [batch, d_model]
            
            # 1. Prediction (What token does the model expect?)
            S_flat = S_t.view(batch, -1)
            hat_x_t = self.pred_net(S_flat)
            
            # 2. Prediction-Error Gate (How surprised is the model?)
            surprise = torch.abs(x_t - hat_x_t)
            gate_logits = self.gate_proj(surprise)
            
            # Modulate with inertia: g_t = sigmoid(surprise - (base_inertia + velocity))
            threshold = self.base_inertia.unsqueeze(0) + self.inertia_scale * V_t
            g_t = torch.sigmoid(gate_logits - threshold) # [batch, num_slots]
            g_t_expanded = g_t.unsqueeze(-1)
            
            # 3. Slot Update
            x_t_expanded = x_t.unsqueeze(1).expand(-1, self.num_slots, -1)
            update_input = torch.cat([S_t, x_t_expanded], dim=-1)
            U_t = self.update_net(update_input)
            
            S_prime = g_t_expanded * U_t + (1 - g_t_expanded) * S_t
            
            # 4. Dynamic Relational Wave Propagation (Cross-Attention between slots)
            q = self.wq(S_prime)
            k = self.wk(S_prime)
            v = self.wv(S_prime)
            
            attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5), dim=-1)
            wave_update = torch.bmm(attn_weights, v)
            
            S_new = self.wave_norm(S_prime + wave_update)
            
            # 5. Update Velocity (Inertia)
            delta_S = torch.norm(S_new - S_t, dim=-1) # [batch, num_slots]
            V_t = self.inertia_momentum * V_t + (1 - self.inertia_momentum) * delta_S
            
            S_t = S_new
            
            # 6. Predict Next Token Logits
            next_logits = self.out_proj(S_t.view(batch, -1))
            
            gates.append(g_t)
            logits.append(next_logits)
            slots.append(S_t)
            
        gates_tensor = torch.stack(gates, dim=1) # [batch, seq, num_slots]
        logits_tensor = torch.stack(logits, dim=1) # [batch, seq, vocab]
        slots_tensor = torch.stack(slots, dim=1) # [batch, seq, num_slots, d_model]
        
        return logits_tensor, gates_tensor, slots_tensor

# ==========================================
# 2. Main Training and Benchmark Script
# ==========================================

def run_campaign(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Device: {device} | Total GPUs Detected: {num_gpus}")
    
    # 1. Load GPT-2 Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # 2. Load and Prepare Dataset (Wikitext-2)
    print("Downloading Wikitext-2 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    # Simple tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.seq_len, padding="max_length")
        
    print("Tokenizing dataset...")
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 3. Instantiate Model
    print(f"Initializing PSSA Language Model (d_model={args.d_model}, slots={args.num_slots})...")
    model = PSSALanguageModel(vocab_size=vocab_size, d_model=args.d_model, num_slots=args.num_slots)
    
    # Multi-GPU support wrapping
    if num_gpus > 1:
        print(f"Wrapping model in nn.DataParallel across {num_gpus} GPUs.")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.2f} Million")
    
    # 4. Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    history = {
        "step": [],
        "ce_loss": [],
        "decorr_loss": [],
        "total_loss": [],
        "sparsity": [],
        "tokens_per_sec": []
    }
    
    step_count = 0
    start_time = time.time()
    
    print("\n--- Starting PSSA-LM Training Campaign ---")
    for epoch in range(args.epochs):
        if step_count >= args.max_steps:
            break
            
        for batch in dataloader:
            if step_count >= args.max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            # Targets are shifted right
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            # Forward pass with mixed-precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, gates, slots = model(inputs)
                
                # Compute Cross-Entropy Loss
                loss_ce = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=tokenizer.pad_token_id)
                
                # Compute Slot Decorrelation Loss (to enforce slot independence/entropy)
                flat_slots = slots.view(-1, args.num_slots, args.d_model)
                slot_sims = []
                for i in range(args.num_slots):
                    for j in range(i+1, args.num_slots):
                        sim = F.cosine_similarity(flat_slots[:, i, :], flat_slots[:, j, :], dim=-1)
                        slot_sims.append(sim.pow(2).mean())
                loss_decorr = sum(slot_sims) if len(slot_sims) > 0 else torch.tensor(0.0, device=device)
                
                # Total Loss
                total_loss = loss_ce + args.lambda_decorr * loss_decorr
                
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            step_count += 1
            
            # Telemetry tracking
            if step_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_processed = args.batch_size * args.seq_len * args.log_interval
                throughput = tokens_processed / elapsed
                
                # Sparsity computation (proportion of gates < 0.2)
                sparse_pct = (gates < 0.2).float().mean().item() * 100
                
                print(f"Step {step_count:4d}/{args.max_steps:4d} | "
                      f"CE Loss: {loss_ce.item():.4f} | "
                      f"Decorr Loss: {loss_decorr.item():.4f} | "
                      f"Sparsity: {sparse_pct:.1f}% | "
                      f"Speed: {throughput:.0f} tok/sec")
                      
                history["step"].append(step_count)
                history["ce_loss"].append(loss_ce.item())
                history["decorr_loss"].append(loss_decorr.item())
                history["total_loss"].append(total_loss.item())
                history["sparsity"].append(sparse_pct)
                history["tokens_per_sec"].append(throughput)
                
                start_time = time.time()
                
            # Periodic checkpoint saving and Hugging Face upload
            if step_count % args.save_interval == 0:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/pssa_llm_kaggle_step_{step_count}.pth"
                torch.save(
                    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    checkpoint_path
                )
                print(f"💾 Checkpoint saved locally at step {step_count}")
                
                if args.hf_repo and args.hf_token:
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        print(f"📡 Uploading checkpoint to Hugging Face repository '{args.hf_repo}' at step {step_count}...", flush=True)
                        api.upload_file(
                            path_or_fileobj=checkpoint_path,
                            path_in_repo=f"pssa_llm_kaggle_step_{step_count}.pth",
                            repo_id=args.hf_repo,
                            repo_type="model",
                            token=args.hf_token
                        )
                        print("✅ Checkpoint upload complete!", flush=True)
                    except Exception as e:
                        print(f"[HF WARNING] Failed to upload checkpoint at step {step_count}: {e}", flush=True)
                        
    # 5. Generate plots

    print("\nTraining completed! Compiling figures...")
    os.makedirs("results", exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cross Entropy Loss', color=color)
    ax1.plot(history["step"], history["ce_loss"], color=color, label="CE Loss", linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Sparsity (% of Gates < 0.2)', color=color)
    ax2.plot(history["step"], history["sparsity"], color=color, label="Gate Sparsity", linewidth=2, linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('PSSA-LM Gated Training Performance (Kaggle T4)')
    fig.tight_layout()  
    plt.savefig("results/pssa_lm_training_curves.png", dpi=150)
    plt.close()
    print("Saved training curves to results/pssa_lm_training_curves.png")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/pssa_llm_kaggle.pth"
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        checkpoint_path
    )
    print(f"Model saved to {checkpoint_path}")
    
    # Upload to Hugging Face
    if args.hf_repo and args.hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            print(f"📡 Uploading final checkpoint to Hugging Face repository '{args.hf_repo}'...", flush=True)
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo="pssa_llm_kaggle.pth",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token
            )
            print("✅ Final Hugging Face upload complete!", flush=True)
        except Exception as e:
            print(f"[HF WARNING] Failed to upload final checkpoint: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSSA-LM Kaggle Training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--seq_len", type=int, default=128, help="Context sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--num_slots", type=int, default=5, help="Number of persistent memory slots")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda_decorr", type=float, default=0.2, help="Slot decorrelation weight")
    parser.add_argument("--log_interval", type=int, default=25, help="Logging steps interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Steps interval to save and upload checkpoints")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face repository ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--verify_only", action="store_true", help="If set, runs tiny verification run")
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Override to lightweight settings for verification check
        args.max_steps = 2
        args.batch_size = 2
        args.seq_len = 16
        args.d_model = 64
        args.epochs = 1
        args.log_interval = 1
        
    run_campaign(args)

