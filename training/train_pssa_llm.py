import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_llm import PSSALanguageModel

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def train_pssa_lm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 8: PSSA-LM Training on {device}")
    
    # 1. Download and Prepare TinyShakespeare
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print("Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
        
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 2. Initialize Model
    d_model = 128
    num_slots = 5
    model = PSSALanguageModel(vocab_size=vocab_size, d_model=d_model, num_slots=num_slots).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 16
    seq_len = 64
    epochs = 500
    
    print("\n--- Training PSSA-LM ---")
    
    for epoch in range(epochs):
        model.train()
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        optimizer.zero_grad()
        logits, gates, slots = model(x)
        
        # Cross Entropy Loss
        loss_ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        # Slot Decorrelation Loss (Mutual Information Penalty)
        # slots shape: [batch, seq, num_slots, d_model]
        flat_slots = slots.view(-1, num_slots, d_model)
        slot_sims = []
        for i in range(num_slots):
            for j in range(i+1, num_slots):
                sim = F.cosine_similarity(flat_slots[:, i, :], flat_slots[:, j, :], dim=-1)
                slot_sims.append(sim.pow(2).mean())
        loss_decorr = sum(slot_sims) if len(slot_sims) > 0 else torch.tensor(0.0, device=device)
        
        loss = loss_ce + 0.5 * loss_decorr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            # Evaluate Sparsity Benchmark
            mean_gate = gates.mean().item()
            sparse_pct = (gates < 0.2).float().mean().item() * 100
            
            print(f"Iter {epoch+1:4d} | CE Loss: {loss_ce.item():.4f} | Decorr: {loss_decorr.item():.4f}")
            print(f"          -> Mean Gate: {mean_gate:.4f} | Tokens < 0.2: {sparse_pct:.1f}%")
            
    print("\n--- Final Sparsity Benchmark ---")
    if sparse_pct >= 80.0:
        print("✅ PASSED: Target 80-90% of tokens produced gate values < 0.2.")
    else:
        print("❌ FAILED: The model is too dense. Gate values are staying open too often.")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pssa_llm.pth")
    print("PSSA-LM trained and saved.")

if __name__ == "__main__":
    train_pssa_lm()
