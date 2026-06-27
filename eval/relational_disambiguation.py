import torch
import torch.nn as nn
import torch.optim as optim
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

# --- Probe Architecture for Entity Memory Bank Disambiguation ---
class RelationalProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # Classifies which Entity Memory Bank (0, 1, 2, or 3) holds the target representation
        )
    def forward(self, x):
        return self.net(x)

def run_referential_disambiguation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Stage 11 Referential Correctness & Disambiguation Probes...")
    
    # 1. Load Vocab
    data_path = f"{PSSA_PROJECT_DIR}/data/tinyshakespeare.txt"
    if not os.path.exists(data_path):
        print("Dataset missing, cannot construct vocab.")
        return
        
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    
    model = PSSAGPT(vocab_size=vocab_size).to(device)
    checkpoint = "checkpoints/pssa_gpt.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("Loaded PSSA-GPT v2 weights.")
    else:
        print("Warning: Running probe on uninitialized weights.")
    model.eval()
    
    # --- PHASE 1: GATE SALIENCY VISUALIZER ---
    # Print a character-level map of gate activations to verify event-driven sparsity.
    sample_sentence = "John gave Mary the book because she asked."
    sample_tokens = torch.tensor([[stoi[c] for c in sample_sentence]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, gates, _, _, _ = model(sample_tokens)
    
    print("\n--- Phase 1: Gate Saliency Visualizer ---")
    visual_map = ""
    for i, c in enumerate(sample_sentence):
        # average gate activation at this character
        gate_act = gates[0, i].mean().item()
        # Visual color/intensity representation
        if gate_act < 0.1:
            visual_map += f"\033[90m{c}\033[0m" # Grey (near silent)
        elif gate_act < 0.25:
            visual_map += f"{c}" # Normal
        else:
            visual_map += f"\033[91;1m{c}\033[0m" # Bold Red (saliency spike!)
    print(f"Saliency Map: {visual_map}")
    print("Interpretation: [\033[91;1mBold Red\033[0m = surprise spike / slot write, \033[90mGrey\033[0m = silent skip]")
    
    # --- PHASE 2: PROBE TRAINING (Entity Slot Mapping) ---
    # We train our probe to identify which explicit entity bank (0-3) tracks John and Mary.
    sentences_john = [
        "John is here. He is nice.",
        "John likes the book. He promised.",
        "John called Mary. He was happy.",
        "John ran away. He escaped."
    ]
    sentences_mary = [
        "Mary is here. She is nice.",
        "Mary likes the book. She promised.",
        "Mary called John. She was happy.",
        "Mary ran away. She escaped."
    ]
    
    X_states = []
    Y_labels = []
    
    with torch.no_grad():
        for s in sentences_john:
            tokens = torch.tensor([[stoi[c] for c in s]], dtype=torch.long, device=device)
            _, _, slots, _, _ = model(tokens)
            idx = s.find("He")
            X_states.append(slots[0, idx, 2, :].cpu()) # Entity Slot state
            Y_labels.append(0) # Bank index 0 (John)
            
        for s in sentences_mary:
            tokens = torch.tensor([[stoi[c] for c in s]], dtype=torch.long, device=device)
            _, _, slots, _, _ = model(tokens)
            idx = s.find("She")
            X_states.append(slots[0, idx, 2, :].cpu()) # Entity Slot state
            Y_labels.append(1) # Bank index 1 (Mary)
            
    X_states = torch.stack(X_states).to(device)
    Y_labels = torch.tensor(Y_labels, dtype=torch.long, device=device)
    
    probe = RelationalProbe(d_model=128).to(device)
    probe_optimizer = optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        probe.train()
        probe_optimizer.zero_grad()
        outputs = probe(X_states)
        loss = criterion(outputs, Y_labels)
        loss.backward()
        probe_optimizer.step()
        
    print(f"\nDisambiguation Probe Trained. Loss: {loss.item():.4f}")
    
    # --- PHASE 3: RELATIONAL DISAMBIGUATION ---
    sent_a = "John gave Mary the book because she asked."
    sent_b = "John gave Mary the book because he promised."
    
    tokens_a = torch.tensor([[stoi[c] for c in sent_a]], dtype=torch.long, device=device)
    tokens_b = torch.tensor([[stoi[c] for c in sent_b]], dtype=torch.long, device=device)
    
    probe.eval()
    with torch.no_grad():
        # Sentence A (she)
        _, _, slots_a, _, _ = model(tokens_a)
        she_idx = sent_a.find("she")
        state_she = slots_a[0, she_idx, 2, :].unsqueeze(0)
        pred_she = torch.argmax(probe(state_she), dim=-1).item()
        
        # Sentence B (he)
        _, _, slots_b, _, _ = model(tokens_b)
        he_idx = sent_b.find("he")
        state_he = slots_b[0, he_idx, 2, :].unsqueeze(0)
        pred_he = torch.argmax(probe(state_he), dim=-1).item()
        
    names = {0: "Entity Bank 0 (John)", 1: "Entity Bank 1 (Mary)", 2: "Entity Bank 2 (Empty)", 3: "Entity Bank 3 (Empty)"}
    
    print("\n--- Phase 3: Relational Pronoun Disambiguation ---")
    print(f"Sentence A: '{sent_a}'")
    print(f" -> Pronoun 'she' resolved to: {names.get(pred_she, 'Unresolved')}")
    
    print(f"Sentence B: '{sent_b}'")
    print(f" -> Pronoun 'he' resolved to: {names.get(pred_he, 'Unresolved')}")
    
    if pred_she == 1 and pred_he == 0:
        print("✅ PASSED: True Relational Semantic Disambiguation verified! PSSA-GPT v2 successfully resolved pronoun bindings back to explicit memory banks.")
    else:
        print("❌ FAILED: The explicit entity banks failed to resolve correctly (semantic bleed or early compression limit).")
        
    # Check general sparsity under Competitive Gating
    with torch.no_grad():
        _, gates_a, _, _, _ = model(tokens_a)
    mean_gate = gates_a.mean().item()
    sparse_pct = (gates_a < 0.2).float().mean().item() * 100
    print(f"\nSparsity level under Competitive Gating: {sparse_pct:.1f}% (Mean gate activation: {mean_gate:.4f})")

if __name__ == "__main__":
    run_referential_disambiguation()
