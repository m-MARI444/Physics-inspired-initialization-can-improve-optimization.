import torch
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSACausal

def run_counterfactual():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 4: Counterfactual Simulation Benchmark")
    
    traj_path = f"{PSSA_PROJECT_DIR}/data/arm_trajectory.pt"
    act_path  = f"{PSSA_PROJECT_DIR}/data/causal_actions.pt"
    
    trajectory = torch.load(traj_path, map_location=device)
    actions = torch.load(act_path, map_location=device)
    
    traj_mean = trajectory.mean(dim=1, keepdim=True)
    traj_std = trajectory.std(dim=1, keepdim=True) + 1e-6
    act_mean = actions.mean(dim=1, keepdim=True)
    act_std = actions.std(dim=1, keepdim=True) + 1e-6
    
    norm_trajectory = (trajectory - traj_mean) / traj_std
    norm_actions = (actions - act_mean) / act_std
    
    model = PSSACausal(d_in=6, d_action=6, d_model=64, d_out=6, vocab_size=8).to(device)
    model.load_state_dict(torch.load(f"{PSSA_PROJECT_DIR}/checkpoints/pssa_v2_causal.pth", map_location=device))
    model.eval()
    
    print("\n[Phase 1] Normal Operation (0-500 steps)")
    half_idx = min(500, norm_trajectory.shape[1] // 2)
    x_in_normal = norm_trajectory[:, :half_idx, :]
    a_in_normal = norm_actions[:, :half_idx, :]
    
    print("[Phase 2] The Blindfold Counterfactual Injection")
    # We turn off physical sensors (x_t = 0)
    blind_steps = 100
    x_in_blind = torch.zeros((1, blind_steps, 6), device=device)
    
    # We inject a hypothetical action: commanding base (joint 0) to rotate left
    a_in_hypo = torch.zeros((1, blind_steps, 6), device=device)
    a_in_hypo[:, :, 0] = 5.0 # Large base left command
    
    test_x = torch.cat([x_in_normal, x_in_blind], dim=1)
    test_a = torch.cat([a_in_normal, a_in_hypo], dim=1)
    
    with torch.no_grad():
        _, preds_lang, _, _ = model(test_x, test_a)
        
        hypo_lang_logits = preds_lang[:, half_idx:, :]
        hypo_lang_preds = torch.argmax(hypo_lang_logits, dim=-1)
        
        # Vocab mapping: 2 = "base left"
        base_left_token = 2
        
        predicted_tokens = hypo_lang_preds[0, :].tolist()
        success_rate = predicted_tokens.count(base_left_token) / blind_steps
        
        print(f"Counterfactual Action injected: 'Command base left during blindness'")
        print(f"Model internal semantic simulation output matches 'base left': {success_rate*100:.2f}%")
        
        if success_rate > 0.8:
            print("✅ PASSED: PSSA successfully simulated a counterfactual reality without visual input!")
        else:
            print("❌ FAILED: The model failed to causalize the injected action.")

if __name__ == "__main__":
    run_counterfactual()
