import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

def run_interpretability_dashboard():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("======================================================================")
    print("            PSSA-GPT v2 COGNITIVE & INTERPRETABILITY DASHBOARD        ")
    print("======================================================================\n")
    
    # 1. Load Vocab and Model
    vocab_path = "checkpoints/pssa_gpt_vocab.pth"
    checkpoint = "checkpoints/pssa_gpt.pth"
    if not os.path.exists(vocab_path) or not os.path.exists(checkpoint):
        print("Model or vocab checkpoint missing. Run training first.")
        return
        
    stoi, itos = torch.load(vocab_path)
    vocab_size = len(stoi)
    
    model = PSSAGPT(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    # --- TEST CASES: COUNTERFACTUAL PAIRS ---
    sentences = [
        ("john gave mary the book because she asked .", "she", "mary"),
        ("john gave mary the book because he promised .", "he", "john")
    ]
    
    slot_names = {0: "Syntax", 1: "Semantic", 2: "Entity", 3: "Discourse", 4: "World"}
    
    for idx, (sent, pronoun, target_referent) in enumerate(sentences):
        words = sent.split()
        tokens = torch.tensor([[stoi[w] for w in words]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, gates, slots, adj, entities = model(tokens)
            
        print(f"\nEvaluating Sentence {idx+1}: \"{sent}\"")
        print("-" * 70)
        
        # 1. GATE SURPRISE VISUALIZER
        print("1. Event-Driven Gate Saliency Map:")
        visual_map = ""
        for t, w in enumerate(words):
            gate_act = gates[0, t].mean().item()
            if gate_act < 0.1:
                visual_map += f"\033[90m{w}\033[0m "
            elif gate_act < 0.25:
                visual_map += f"{w} "
            else:
                visual_map += f"\033[91;1m{w}\033[0m "
        print(f"   {visual_map}")
        print("   [\033[91;1mBold Red\033[0m = surprise write, \033[90mGrey\033[0m = silent skip]")
        
        # 2. DYNAMIC SPARSE GRAPH ATTENTION CONNECTIONS
        pron_step = words.index(pronoun)
        A_pron = adj[0, pron_step] # [5, 5]
        
        print(f"\n2. Dynamic Sparse Graph Topology at pronoun '{pronoun}':")
        for s_idx in range(5):
            active_links = [slot_names[j] for j in range(5) if A_pron[s_idx, j] > 0.0]
            print(f"   Slot {slot_names[s_idx]:9s} -> connected to -> {active_links}")
            
        # 3. EXPLICIT ENTITY MEMORY RESOLUTION
        state_pron = slots[0, pron_step, 2, :] # [d_model]
        banks_pron = entities[0, pron_step, :, :] # [4, d_model]
        
        # In Anchored Dynamic Routing:
        # Bank 0 tracks the first entity (index 0)
        # Bank 1 tracks the second entity (index 2)
        target_bank_idx = 0 if target_referent == words[0] else 1
        
        # Calculate cosine similarities against pronoun state
        similarities = F.cosine_similarity(state_pron.unsqueeze(0), banks_pron, dim=-1)
        resolved_bank = torch.argmax(similarities).item()
        
        print(f"\n3. Explicit Entity Memory Bank registers at '{pronoun}':")
        for b_idx in range(4):
            content = words[0] if b_idx == 0 else words[2] if b_idx == 1 else "empty"
            marker = ""
            if b_idx == resolved_bank:
                marker += "<- [RESOLVED REFERENCE] "
            if b_idx == target_bank_idx:
                marker += "<- [GROUND-TRUTH TARGET]"
            print(f"   Entity Bank {b_idx}: tracks \"{content:5s}\" {marker}")
            
        if resolved_bank == target_bank_idx:
            print(f"\n🎉 SUCCESS: Pronoun '{pronoun}' resolved correctly to \"{target_referent}\"!")
        else:
            print(f"\n❌ FAILED: Resolved pronoun to bank {resolved_bank} instead of target bank {target_bank_idx}.")
            
    print("\n======================================================================\n")

if __name__ == "__main__":
    run_interpretability_dashboard()
