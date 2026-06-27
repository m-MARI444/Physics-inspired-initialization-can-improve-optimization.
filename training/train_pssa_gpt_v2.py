import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT
from training.pssa_telemetry import PSSATelemetry

def get_fineweb_edu_dataloader(tokenizer, batch_size=4, seq_len=128):
    """Streams FineWeb-Edu for internet-scale cognition training."""
    print("Loading FineWeb-Edu stream...")
    # Streaming dataset directly from HuggingFace
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    def data_generator():
        buffer_ids = []
        for example in dataset:
            tokens = tokenizer.encode(example['text'])
            buffer_ids.extend(tokens)
            
            # Yield full seq_len chunks
            while len(buffer_ids) >= seq_len + 1:
                chunk_in = buffer_ids[:seq_len]
                chunk_tgt = buffer_ids[1:seq_len+1]
                yield torch.tensor(chunk_in, dtype=torch.long), torch.tensor(chunk_tgt, dtype=torch.long)
                buffer_ids = buffer_ids[seq_len:] # Non-overlapping to process more data faster
                
    def batch_generator():
        gen = data_generator()
        while True:
            batch_inputs = []
            batch_targets = []
            for _ in range(batch_size):
                try:
                    inp, tgt = next(gen)
                    batch_inputs.append(inp)
                    batch_targets.append(tgt)
                except StopIteration:
                    break
            if len(batch_inputs) == 0:
                break
            yield torch.stack(batch_inputs), torch.stack(batch_targets)

    return batch_generator()

def train_pssa_gpt_scaled():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 35.5: Internet-Scale Sparse Cognition Validation on {device}", flush=True)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = len(tokenizer) # 50257
    print(f"Full GPT-2 Vocabulary loaded: {vocab_size} tokens.", flush=True)
    
    # --- STAGE 35.5: PROGRESSIVE PARAMETER SCALING ---
    # Upgraded from (d=64, L=3, slots=5) -> (d=256, L=6, slots=8)
    d_model = 256
    num_layers = 6
    num_slots = 8
    
    print(f"Initializing ~50M Parameter PSSA Model (d_model={d_model}, layers={num_layers}, slots={num_slots})...")
    model = PSSAGPT(vocab_size=vocab_size, d_model=d_model, num_slots=num_slots, tau=0.15, num_scopes=3, num_layers=num_layers).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()
    
    # Initialize Stage 35.5 Massive Telemetry System
    config = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_slots": num_slots,
        "dataset": "FineWeb-Edu",
        "batch_size": 1,
        "seq_len": 128
    }
    telemetry = PSSATelemetry(project_name="pssa_scaling", run_name="stage_35.5_fineweb", config=config, use_wandb=False) # Keep local for sandbox
    
    dataloader = get_fineweb_edu_dataloader(tokenizer, batch_size=config["batch_size"], seq_len=config["seq_len"])
    
    # Training Loop
    model.train()
    total_steps = 100 # Short run to validate the pipeline
    
    print("\n--- Starting Massive Instrumentation Scaling Run ---", flush=True)
    
    start_time = time.time()
    for step, (input_ids, target_ids) in enumerate(dataloader):
        if step >= total_steps:
            break
            
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        # Sparse routing creates complex computation graphs, autocast saves VRAM
        with autocast(enabled=(device.type == "cuda")):
            # Note: We omit actor_mask for FineWeb since we want generalized reasoning
            outputs = model(input_ids, actor_mask=None)
            logits = outputs[0]
            recon_loss = outputs[9]
            
            loss_ce = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Simplified scaling loss (CE + 10% Recon)
            loss = loss_ce + 0.10 * recon_loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Push 20+ variables to the telemetry system
        telemetry.log_step(loss.item(), outputs)
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | CE: {loss_ce.item():.4f} | Recon: {recon_loss.item():.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()
            
    telemetry.finish()
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pssa_gpt_scaled.pth")
    print("Stage 35.5 Scaled model saved.", flush=True)

if __name__ == "__main__":
    train_pssa_gpt_scaled()
