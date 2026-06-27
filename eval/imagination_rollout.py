import torch
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator

def run_imagination():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 4.5: Imagination Rollout Stability Benchmark")
    
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
    
    model = PSSASimulator(d_in=6, d_action=6, d_model=64, d_out=6, vocab_size=8).to(device)
    model.load_state_dict(torch.load(f"{PSSA_PROJECT_DIR}/checkpoints/pssa_v2_simulator.pth", map_location=device))
    model.eval()
    
    # 1. Observe 500 steps
    half_idx = min(500, norm_trajectory.shape[1] // 2)
    x_in_normal = norm_trajectory[:, :half_idx, :]
    a_in_normal = norm_actions[:, :half_idx, :]
    
    with torch.no_grad():
        # Prime the persistent state
        _, _, states, _ = model(x_in_normal, a_in_normal, blindfold_start_idx=None)
        persistent_state = states[:, -1:, :] # The state exactly at step 500
        
        # 2. Blindfold indefinitely, feed action sequence only
        blind_steps = 100
        x_in_blind = torch.zeros((1, blind_steps, 6), device=device) # Ignored entirely
        
        # Inject pure actions (e.g., base left)
        a_in_hypo = torch.zeros((1, blind_steps, 6), device=device)
        a_in_hypo[:, :, 0] = 5.0 # Large base left command
        
        # Run dynamics ONLY
        _, preds_lang, _, _ = model(x_in_blind, a_in_hypo, prev_state=persistent_state, blindfold_start_idx=0)
        
        hypo_lang_preds = torch.argmax(preds_lang, dim=-1)
        base_left_token = 2
        
        predicted_tokens = hypo_lang_preds[0, :].tolist()
        print(f"Predicted Tokens: {predicted_tokens[:20]}")
        success_rate = predicted_tokens.count(base_left_token) / blind_steps
        
        print(f"Action sequence injected: 'Command base left for 100 steps'")
        print(f"Imagination language output matching 'base left': {success_rate*100:.2f}%")
        
        if success_rate > 0.8:
            print("✅ PASSED: True Causal Simulation! PSSA successfully dreamed a coherent future sequence purely from latent dynamics.")
        else:
            print("❌ FAILED: The imagination collapsed or hallucinated randomly.")

if __name__ == "__main__":
    run_imagination()
