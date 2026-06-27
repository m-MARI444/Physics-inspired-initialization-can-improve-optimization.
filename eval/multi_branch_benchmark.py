import torch
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator

def run_multi_branch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 5: Multi-Branch Blindfold Benchmark")
    
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
        _, _, states, _ = model(x_in_normal, a_in_normal, blindfold_start_idx=None)
        persistent_state = states[:, -1:, :]
        
        blind_steps = 100
        x_in_blind = torch.zeros((1, blind_steps, 6), device=device)
        
        # UNIVERSE A: Command Base Left
        a_in_left = torch.zeros((1, blind_steps, 6), device=device)
        a_in_left[:, :, 0] = 5.0 # Base Left
        
        _, preds_lang_a, states_a, _ = model(x_in_blind, a_in_left, prev_state=persistent_state, blindfold_start_idx=0)
        
        # UNIVERSE B: Command Base Right
        a_in_right = torch.zeros((1, blind_steps, 6), device=device)
        a_in_right[:, :, 0] = -5.0 # Base Right
        
        _, preds_lang_b, states_b, _ = model(x_in_blind, a_in_right, prev_state=persistent_state, blindfold_start_idx=0)
        
        # Evaluate divergence
        latent_divergence = torch.mean((states_a - states_b)**2).item()
        
        tokens_a = torch.argmax(preds_lang_a, dim=-1)[0, :].tolist()
        tokens_b = torch.argmax(preds_lang_b, dim=-1)[0, :].tolist()
        
        print("\n--- Multi-Branch Evaluation ---")
        print(f"Latent State Divergence between universes: {latent_divergence:.4f}")
        print(f"Universe A Tokens (Move Left):  {tokens_a[:20]}")
        print(f"Universe B Tokens (Move Right): {tokens_b[:20]}")
        
        if latent_divergence > 0.05:
            print("✅ PASSED: The latent states successfully diverged based on the counterfactual actions!")
        else:
            print("❌ FAILED: The model suffered from deterministic mode collapse again.")
            
if __name__ == "__main__":
    run_multi_branch()
