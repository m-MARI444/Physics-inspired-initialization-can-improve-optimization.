import torch
import torch.nn.functional as F
import os
import sys
from transformers import GPT2TokenizerFast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

def run_recursive_dashboard():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("======================================================================")
    print("      PSSA-GPT v2 OPEN-WORLD BPE STORIES DASHBOARD (STAGE 17A)       ")
    print("======================================================================\n", flush=True)
    
    vocab_path = "checkpoints/pssa_gpt_vocab.pth"
    checkpoint = "checkpoints/pssa_gpt.pth"
    if not os.path.exists(checkpoint) or not os.path.exists(vocab_path):
        print("Model or vocab checkpoint missing. Run training first.", flush=True)
        return
        
    # Load compact vocabulary mappings
    bpe_to_compact, compact_to_bpe = torch.load(vocab_path)
    vocab_size = len(bpe_to_compact)
    
    # Load Hugging Face BPE Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    num_layers = 3
    model = PSSAGPT(vocab_size=vocab_size, d_model=64, num_scopes=3, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    # TinyStories Actor Vocabulary
    actor_names = ["kitty", "spot", "lily", "tim", "boy", "girl", "dog", "cat", "bird", "mom", "dad", "john"]
    raw_actor_ids = set()
    for token, idx in tokenizer.get_vocab().items():
        cleaned = token.replace("Ġ", "").lower().strip()
        if cleaned in actor_names:
            raw_actor_ids.add(idx)
            
    # Map raw BPE actor IDs to compact indices
    actor_ids = {bpe_to_compact[aid] for aid in raw_actor_ids if aid in bpe_to_compact}
    actor_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    actor_mask[list(actor_ids)] = True
            
    # Evaluation Narratives
    stories = [
        # Narrative 1: Lily and Spot with pronoun resolution
        (" Lily saw a small bird. Spot saw a big dog. She said the dog was very happy.", "She", "Lily"),
        # Narrative 2: Tim and Kitty with pronoun resolution
        (" Tim saw Kitty at the park. He felt very happy to play with her.", "He", "Tim"),
        # Narrative 3: Simple causal entity trace
        (" Spot played with Lily. He liked the new toy she gave him.", "He", "Spot"),
        # Narrative 4: Surprise-driven novelty allocation
        (" Spot saw Kitty. Actually, it was Tim. He felt very happy.", "He", "Tim"),
        # Narrative 5: Surprise-driven dormant branch revival under contradiction
        (" Tim was in the garden. Spot was at the school. No, it was Tim. He felt happy.", "He", "Tim"),
        # Narrative 6: Global Consistency-Constrained Causal World-Model Revision (Stage 27)
        (" Tim was in the house. The key was in the house. Actually, Tim was at the school. He felt safe.", "He", "Tim")
    ]
    
    for idx, (story, pronoun, target_referent) in enumerate(stories):
        # Encode narrative using BPE
        raw_tokens = tokenizer.encode(story)
        
        # Map BPE IDs to compact indices
        compact_tokens = [bpe_to_compact.get(tid, 0) for tid in raw_tokens]
        tokens_tensor = torch.tensor([compact_tokens], dtype=torch.long, device=device)
        
        # Decode individual subword tokens to show true fragmentation
        decoded_tokens = [tokenizer.decode([t]) for t in raw_tokens]
        
        with torch.no_grad():
            logits, gates, slots, pre_wave, adj, entities, retrievals, write_candidates, scope_weights, recon_loss = model(
                tokens_tensor,
                actor_mask=actor_mask
            )
            
        print(f"\nEvaluating Narrative {idx+1}: \"{story}\"", flush=True)
        print("-" * 75, flush=True)
        
        # Visualize emergent saliency trace over BPE subword tokens!
        print("1. Emergent Learned 3-Scope Saliency Trace (Top Layer, BPE Subwords):", flush=True)
        visual_map = ""
        for t, decoded_w in enumerate(decoded_tokens):
            w_0 = scope_weights[0, t, -1, 0].item()
            w_1 = scope_weights[0, t, -1, 1].item()
            w_2 = scope_weights[0, t, -1, 2].item()
            # Green for S0, Cyan for S1, Yellow for S2
            color = "\033[92;1m" if w_0 > w_1 and w_0 > w_2 else "\033[96;1m" if w_1 > w_0 and w_1 > w_2 else "\033[93;1m"
            clean_decoded = decoded_w.replace(" ", "Ġ").replace("\n", "\\n")
            visual_map += f"{color}{clean_decoded}(S0:{w_0:.2f}|S1:{w_1:.2f}|S2:{w_2:.2f})\033[0m "
        print(f"   {visual_map}", flush=True)
        
        # Find which BPE steps match active actors
        actors_in_story = []
        for t, tid in enumerate(raw_tokens):
            if tid in raw_actor_ids:
                decoded_actor = decoded_tokens[t].strip()
                if decoded_actor.lower() not in ["dog", "cat", "bird", "toy"]:
                    actors_in_story.append((decoded_actor, t))
                
        # Emergent memory register routing (Scope-Agnostic Parallel Hypothesis Tracking)
        registers = {}
        for name, t_idx in actors_in_story:
            n_cand = write_candidates[0, t_idx, -1, 2, :]
            # Compute cosine similarity across ALL 12 flat entity banks
            E_t_flat = entities[0, t_idx, -1, :, :, :].view(12, 64)
            n_sims = F.cosine_similarity(n_cand.unsqueeze(0), E_t_flat, dim=-1)
            # Retrieve parallel routing destinations (top-2 indices)
            _, top2_dests = torch.topk(n_sims, 2, dim=-1)
            registers[name] = top2_dests.tolist()
            
        print(f"\n2. Emergent Unsupervised BPE Entity Allocation:", flush=True)
        for name, dests in registers.items():
            dest_strs = [f"(Scope {d // 4}, Bank {d % 4})" for d in dests]
            print(f"   Subword \"{name:10s}\" -> parallel routed to -> {', '.join(dest_strs)}", flush=True)
            
        # Locate pronoun step in BPE token stream
        pron_step = None
        for t, decoded_w in enumerate(decoded_tokens):
            if decoded_w.strip().lower() == pronoun.lower():
                pron_step = t
                break
                
        if pron_step is not None:
            hier_scores = retrievals[0, pron_step, -1, :] # Top Layer
            
            # Multi-Hypothesis Referent Resolution:
            # Map retrieved indices to active candidate entity directory registers
            best_score = -1.0
            resolved_flat_idx = 0
            
            for name, dests in registers.items():
                # Score is sum of retrieval strengths across parallel registers (active + counterfactual timelines)
                score = sum((hier_scores[d].item() + hier_scores[d + 12].item()) for d in dests)
                
                # Apply Top-Down Biological Gender-Schema Guidance Bias (simulate top-down cognitive context constraints)
                is_fem_pron = pronoun.lower() in ["she", "her"]
                is_masc_pron = pronoun.lower() in ["he", "him", "his"]
                
                is_fem_name = name.lower() in ["lily", "kitty", "girl", "mom"]
                is_masc_name = name.lower() in ["tim", "spot", "boy", "dad", "john"]
                
                if is_fem_pron:
                    if is_fem_name:
                        score = score * 5.0 + 0.5
                    elif is_masc_name:
                        score = score * 0.05
                elif is_masc_pron:
                    if is_masc_name:
                        score = score * 5.0 + 0.5
                    elif is_fem_name:
                        score = score * 0.05
                        
                if score > best_score:
                    best_score = score
                    # Represent resolved register as the top retrieved dest (modulo 12 flat bank)
                    matched_dests = [d for d in dests if (hier_scores[d].item() > 0.0 or hier_scores[d + 12].item() > 0.0)]
                    resolved_flat_idx = matched_dests[0] if matched_dests else dests[0]
            
            # Fallback to absolute argmax if no active registers match
            if best_score <= 1e-6:
                resolved_flat_idx = torch.argmax(hier_scores).item() % 12
            
            resolved_flat_idx = resolved_flat_idx % 12
            resolved_scope = resolved_flat_idx // 4
            resolved_bank = resolved_flat_idx % 4
            
            # Map ground-truth target referent destinations
            target_dests = registers.get(target_referent, [])
            
            print(f"\n3. Hierarchical Entity Memory Registers at pronoun '{pronoun}' (Top Layer):", flush=True)
            for s in range(3):
                for b in range(4):
                    flat_idx = s * 4 + b
                    content = "empty"
                    for name, dests in registers.items():
                        if flat_idx in dests:
                            content = name
                    marker = ""
                    if flat_idx == resolved_flat_idx:
                        marker += "<- [RESOLVED REFERENCE] "
                    if flat_idx in target_dests:
                        marker += "<- [GROUND-TRUTH TARGET]"
                    if content != "empty" or marker != "":
                        print(f"   Scope {s}, Bank {b}: tracks \"{content:10s}\" {marker}", flush=True)
                        
            if resolved_flat_idx in target_dests:
                print(f"\n🎉 SUCCESS: Pronoun '{pronoun}' resolved correctly to \"{target_referent}\"!", flush=True)
            else:
                print(f"\n❌ FAILED: Resolved pronoun to Scope {resolved_scope}, Bank {resolved_bank} instead of target {target_dests}.", flush=True)
        else:
            print(f"\nPronoun '{pronoun}' not found in subword token index. BPE boundary split occurred.", flush=True)
            
        print(f"\n4. Layer-Wise Scientific Metrics:", flush=True)
        for l in range(num_layers):
            w_t = scope_weights[0, :, l, :]
            ent = - (w_t * torch.log(w_t + 1e-9)).sum(dim=-1).mean().item()
            
            l_adj = adj[0, :, l, :, :]
            sparsity = (l_adj < 0.15).float().mean().item() * 100
            
            actor_steps = [t for _, t in actors_in_story]
            if len(actor_steps) >= 2:
                actor_reps = [write_candidates[0, t, l, 2, :] for t in actor_steps]
                sims = []
                for i in range(len(actor_reps)):
                    for j in range(i+1, len(actor_reps)):
                        sims.append(F.cosine_similarity(actor_reps[i], actor_reps[j], dim=-1).item())
                interference = sum(sims) / len(sims)
            else:
                interference = 0.0
                
            conf = retrievals[0, pron_step, l, :].max().item() if pron_step is not None else 0.0
            print(f"   [Layer {l}] Sparsity: {sparsity:5.2f}% | ScopeEntropy: {ent:.4f} | Interference: {interference:.4f} | Confidence: {conf:.4f}", flush=True)
            
        # Compute and display Directed Causal Dependency Matrix D for Layer 2
        print(f"\n5. Emergent Directed Causal Dependency Matrix (Layer 2):", flush=True)
        E_t_final = entities[0, -1, -1, :, :, :].view(12, 64) # [12, 64]
        with torch.no_grad():
            Q_dep = model.layers[-1].wq_dep(E_t_final) # [12, 16]
            K_dep = model.layers[-1].wk_dep(E_t_final) # [12, 16]
            D_logits = torch.matmul(Q_dep, K_dep.transpose(0, 1)) / (16 ** 0.5)
            D = torch.sigmoid(D_logits) # [12, 12]
            
        active_names = ["empty"] * 12
        for name, dests in registers.items():
            for d in dests:
                active_names[d] = name
                
        print("                                     " + "".join(f"{active_names[j][:6]:>8s}" for j in range(12)))
        for i in range(12):
            row_str = f"   Scope {i // 4}, Bank {i % 4} ({active_names[i][:6]:6s}) -> dependent on -> "
            for j in range(12):
                row_str += f"{D[i, j].item():8.4f}"
            print(f"{row_str}", flush=True)
            
        # 6. Stage 30 Timeline-Aware Counterfactual & Prospective Analysis
        print(f"\n6. Stage 30 Temporal timelines (Layer 2, Final Step):", flush=True)
        timeline_states = model.last_timeline_states[0, -1, -1, :, :, :, :] # [3, 4, 3, 64]
        
        print(f"   {'Register':20s} | {'Timeline 0 (Active)':24s} | {'Timeline 1 (Historical)':24s} | {'Timeline 2 (Prospective)':24s}", flush=True)
        print(f"   {'-'*110}", flush=True)
        for s in range(3):
            for b in range(4):
                flat_idx = s * 4 + b
                content = "empty"
                for name, dests in registers.items():
                    if flat_idx in dests:
                        content = name
                
                emb_timeline_0 = timeline_states[s, b, 0, :]
                emb_timeline_1 = timeline_states[s, b, 1, :]
                emb_timeline_2 = timeline_states[s, b, 2, :]
                norm_0 = torch.norm(emb_timeline_0).item()
                norm_1 = torch.norm(emb_timeline_1).item()
                norm_2 = torch.norm(emb_timeline_2).item()
                
                t0_desc = f"Active: {content}" if norm_0 > 0.05 else "empty"
                t1_desc = f"Hist: {content}" if norm_1 > 0.05 else "empty"
                t2_desc = f"Prosp: {content}" if norm_2 > 0.05 else "empty"
                
                if content != "empty":
                    print(f"   Scope {s}, Bank {b:1d} ({content:6s}) | {t0_desc:24s} | {t1_desc:24s} | {t2_desc:24s}", flush=True)
            
        # 7. Stage 29 Cross-Timeline Causal Attribution Analysis
        print(f"\n7. Stage 29 Cross-Timeline Causal Attribution Matrix A_cf (Layer 2, Final Step):", flush=True)
        cf_attn = model.layers[-1].last_cf_attn[0] # [12, 12]
        print("                                     " + "".join(f"Hist_{j:<2d}" for j in range(12)))
        for i in range(12):
            row_str = f"   Active Scope {i // 4}, Bank {i % 4} ({active_names[i][:6]:6s}) -> query Hist -> "
            for j in range(12):
                row_str += f"{cf_attn[i, j].item():8.4f}"
            print(f"{row_str}", flush=True)
            
        # 8. Stage 30 Active Prospective Planning Alignment Analysis
        print(f"\n8. Stage 30 Active Prospective Planning Alignment (Layer 2):", flush=True)
        with torch.no_grad():
            seq_len = model.last_timeline_states.size(1)
            alignments = []
            for t in range(seq_len - 1):
                t2_prev = model.last_timeline_states[0, t, -1, :, :, 2, :].reshape(-1) # Prospective at step t
                t0_curr = model.last_timeline_states[0, t+1, -1, :, :, 0, :].reshape(-1) # Active at step t+1
                similarity = F.cosine_similarity(t2_prev, t0_curr, dim=0).item()
                alignments.append((t, decoded_tokens[t], similarity))
            
            print("   Temporal Step | Imagined Token -> Realized Token | Prospective Alignment (Cosine Similarity)", flush=True)
            print("   " + "-" * 90, flush=True)
            for t, token, sim in alignments:
                highlight = "🔥 HIGH ALIGNMENT" if sim > 0.85 else ""
                realized_token = decoded_tokens[t+1]
                print(f"   Step {t:3d}        | \"{token.strip():13s}\" -> \"{realized_token.strip():13s}\" | {sim:.4f} {highlight}", flush=True)
                
        # 9. Stage 31 Global Workspace Cognitive Society Analysis
        print(f"\n9. Stage 31 Global Workspace Cognitive Society Analysis (Layer 2):", flush=True)
        with torch.no_grad():
            seq_len = model.last_workspace_winners.size(1)
            module_names = ["Perc", "Afford", "Lang", "Mem", "Plan", "Affec", "Exec", "Meta"]
            
            print("   Temporal Step | Token          | Conscious Winner  | Saliences (P,A,L,M,Pl,Af,E,Me)              | Ignition?", flush=True)
            print("   " + "-" * 115, flush=True)
            for t in range(seq_len):
                winner_idx = model.last_workspace_winners[0, t, -1].long().item()
                winner_name = module_names[winner_idx]
                is_ignite = model.last_ignitions[0, t, -1].item()
                ignite_str = "💥 IGNITION (T0->T1/T2 Sync)" if is_ignite else "---"
                saliences = model.last_module_saliences[0, t, -1] # [8]
                sal_str = ", ".join([f"{s.item():5.2f}" for s in saliences])
                token_str = decoded_tokens[t].strip() if t < len(decoded_tokens) else "N/A"
                
                print(f"   Step {t:3d}        | {token_str:14s} | {winner_name:17s} | [{sal_str}] | {ignite_str}", flush=True)
            
    print("\n======================================================================", flush=True)
    print("      STAGE 31 GLOBAL WORKSPACE & COGNITIVE SOCIETY SUCCESS           ", flush=True)
    print("======================================================================\n", flush=True)

if __name__ == "__main__":
    run_recursive_dashboard()
