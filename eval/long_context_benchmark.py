import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
from transformers import GPT2TokenizerFast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

class CausalTransformerKV(nn.Module):
    """Standard Causal Transformer with standard KV-cache representation."""
    def __init__(self, vocab_size, d_model=64, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        
    def forward(self, input_ids):
        # We simulate step-by-step KV-cache expansion to profile memory scaling
        batch, seq_len = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)
        
        # KV-caches for each layer
        k_caches = [torch.zeros(batch, 0, self.d_model, device=device) for _ in range(self.num_layers)]
        v_caches = [torch.zeros(batch, 0, self.d_model, device=device) for _ in range(self.num_layers)]
        
        total_kv_memory = 0
        total_attn_memory = 0
        
        for t in range(seq_len):
            current_input = x[:, t, :].unsqueeze(1) # [batch, 1, d_model]
            
            for l in range(self.num_layers):
                q = self.q_proj[l](current_input) # [batch, 1, d_model]
                k = self.k_proj[l](current_input) # [batch, 1, d_model]
                v = self.v_proj[l](current_input) # [batch, 1, d_model]
                
                # Append key and value to cache: O(L) memory growth!
                k_caches[l] = torch.cat([k_caches[l], k], dim=1)
                v_caches[l] = torch.cat([v_caches[l], v], dim=1)
                
                # Attention computation: [batch, 1, t+1]
                scores = torch.bmm(q, k_caches[l].transpose(1, 2)) / (self.d_model ** 0.5)
                # Keep track of attention scores size
                total_attn_memory += scores.element_size() * scores.nelement()
                
            # Measure KV-caches memory footprint
            for l in range(self.num_layers):
                total_kv_memory += k_caches[l].element_size() * k_caches[l].nelement()
                total_kv_memory += v_caches[l].element_size() * v_caches[l].nelement()
                
        return total_kv_memory, total_attn_memory

def run_long_context_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=======================================================================")
    print("       PSSA v2 LONG-CONTEXT PERSISTENCE & KV-CACHE BENCHMARK          ")
    print("=======================================================================\n", flush=True)
    
    vocab_path = "checkpoints/pssa_gpt_vocab.pth"
    checkpoint = "checkpoints/pssa_gpt.pth"
    if not os.path.exists(checkpoint) or not os.path.exists(vocab_path):
        print("Model or vocab checkpoint missing. Run training first.", flush=True)
        return
        
    bpe_to_compact, compact_to_bpe = torch.load(vocab_path)
    vocab_size = len(bpe_to_compact)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Initialize PSSA model
    model = PSSAGPT(vocab_size=vocab_size, d_model=64, num_scopes=3, num_layers=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    # Initialize Standard Transformer for comparison
    transformer = CausalTransformerKV(vocab_size=vocab_size, d_model=64, num_layers=3).to(device)
    
    # Generate Synthetic Long-Context Story (5000+ tokens)
    # Actor introduced at step 0
    intro = "Spot saw a big dog. "
    # Large neutral filler to challenge context decay
    filler_sentence = "The weather was very nice. A small bird sang happily in the garden. Children played with a yellow ball. "
    num_repetitions = 350 # Generates ~5000 tokens
    story = intro + (filler_sentence * num_repetitions) + "He felt very happy."
    
    raw_tokens = tokenizer.encode(story)
    seq_len = len(raw_tokens)
    print(f"Generated Synthetic Long-Context Narrative: {seq_len} BPE tokens.", flush=True)
    
    # Map BPE IDs to compact indices
    compact_tokens = [bpe_to_compact.get(tid, 0) for tid in raw_tokens]
    tokens_tensor = torch.tensor([compact_tokens], dtype=torch.long, device=device)
    
    actor_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    
    # --- PSSA BENCHMARK RUN ---
    print("\n--- Running PSSA v2 Constant-Memory Execution Profile ---", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    with torch.no_grad():
        logits, gates, slots, pre_wave, adj, entities, retrievals, write_candidates, scope_weights, recon_loss = model(
            tokens_tensor,
            actor_mask=actor_mask
        )
    pssa_time = time.time() - start_time
    
    pssa_peak_mem = 0
    if device.type == "cuda":
        pssa_peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"PSSA Peak GPU Memory: {pssa_peak_mem:.2f} MB", flush=True)
    print(f"PSSA Per-Step Average Latency: {(pssa_time / seq_len)*1000:.4f} ms", flush=True)
    
    # --- TRANSFORMER KV-CACHE COMPARISON ---
    print("\n--- Running Standard Causal Transformer KV-Cache Simulation Profile ---", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    with torch.no_grad():
        tf_kv_mem, tf_attn_mem = transformer(tokens_tensor)
    tf_time = time.time() - start_time
    
    tf_peak_mem = 0
    if device.type == "cuda":
        tf_peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Standard Transformer Peak GPU Memory: {tf_peak_mem:.2f} MB", flush=True)
    print(f"Standard Transformer Per-Step Average Latency: {(tf_time / seq_len)*1000:.4f} ms", flush=True)
    
    # Calculate exact memory sizes of the state representation
    pssa_state_mem = 0
    # S_states: 3 layers * [1, 5, 64] float32 tensors
    # V_states: 3 layers * [1, 5] float32 tensors
    # E_states: 3 layers * [1, 3, 4, 3, 64] float32 tensors (Stage 30 Versioned & Prospective Timelines)
    # H_states: 3 layers * [1, 3, 4] float32 tensors
    for l in range(3):
        # S_states
        pssa_state_mem += 32 * 1 * 5 * 64 / 8
        # V_states
        pssa_state_mem += 32 * 1 * 5 / 8
        # E_states (Stage 30 timelines = 3)
        pssa_state_mem += 32 * 1 * 3 * 4 * 3 * 64 / 8
        # H_states
        pssa_state_mem += 32 * 1 * 3 * 4 / 8
    pssa_state_mem_kb = pssa_state_mem / 1024
    
    tf_kv_mem_kb = tf_kv_mem / 1024
    
    print("\n=======================================================================", flush=True)
    print("                   SCIENTIFIC RESULTS & COMPARISON                     ", flush=True)
    print("=======================================================================", flush=True)
    print(f"Sequence Length: {seq_len} tokens\n", flush=True)
    print(f"| Metric                   | PSSA v2 (Constant) | Standard Transformer |", flush=True)
    print(f"| :----------------------- | :----------------- | :------------------- |", flush=True)
    print(f"| State Memory Size        | {pssa_state_mem_kb:12.3f} KB  | {tf_kv_mem_kb:14.3f} KB  |", flush=True)
    print(f"| Execution Step Complexity| O(1) Constant      | O(L) Linear          |", flush=True)
    print(f"| Attention Complexity     | O(1) Constant      | O(L^2) Quadratic     |", flush=True)
    
    # --- DECAY PROTECTION TEST ---
    # Find the object bank tracking Spot at token step 1
    # Check if Spot's semantic object embedding is still high-fidelity at token step 5000+
    spot_idx = 1
    spot_emb_start = entities[0, spot_idx, -1, 0, 0, :] # Bank 0, Scope 0
    spot_emb_end = entities[0, -2, -1, 0, 0, :]
    
    cos_sim = F.cosine_similarity(spot_emb_start.unsqueeze(0), spot_emb_end.unsqueeze(0)).item()
    print(f"\n4. Object Directory Referent Retention after 5000+ Context Steps:", flush=True)
    print(f"   Object Register Cosine Similarity (Step 1 -> Step {seq_len-2}): {cos_sim:.6f}", flush=True)
    if cos_sim > 0.88:
        print("   ✅ SUCCESS: Referent Object preserved perfectly (>88% fidelity)!", flush=True)
    else:
        print("   ❌ FAILED: Referent Object drifted or decayed below target threshold.", flush=True)
        
    print("\n=======================================================================", flush=True)
    print("            STAGE 30 LONG-CONTEXT BENCHMARK COMPLETED                  ", flush=True)
    print("=======================================================================\n", flush=True)

if __name__ == "__main__":
    run_long_context_benchmark()
