import torch
import torch.nn as nn
import torch.optim as optim
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_robot_state import PSSARobot

def plasticity_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Stage 2.5: Adaptive Plasticity Test (Contradictory Dynamics Injection)")
    
    data_path = f"{PSSA_PROJECT_DIR}/data/arm_trajectory.pt"
    trajectory = torch.load(data_path, map_location=device)
    
    traj_mean = trajectory.mean(dim=1, keepdim=True)
    traj_std = trajectory.std(dim=1, keepdim=True) + 1e-6
    norm_trajectory = (trajectory - traj_mean) / traj_std
    
    inputs = norm_trajectory[:, :-1, :]
    targets = norm_trajectory[:, 1:, :]
    seq_len = inputs.shape[1]
    
    model = PSSARobot(d_in=6, d_model=64, d_out=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()
    
    print("Quickly training model...")
    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        preds, new_state, semantic_proj = model(inputs)
        task_loss = mse_loss(preds, targets)
        s_0 = semantic_proj[:, 0, :]
        s_t = semantic_proj[:, -1, :]
        drift_loss = torch.mean((s_t - s_0)**2)
        loss = task_loss + 0.1 * drift_loss
        loss.backward()
        optimizer.step()
        
    print("Model Trained. Executing Shock Test...")
    model.eval()
    with torch.no_grad():
        half_idx = min(500, seq_len // 2)
        visible_inputs = inputs[:, :half_idx, :]
        
        # Blindfold for 200 steps
        blind_steps = 200
        blind_inputs = torch.zeros((1, blind_steps, 6), device=device)
        
        # Teleport Shock for 50 steps (Abrupt change in state)
        shock_steps = 50
        shock_inputs = torch.ones((1, shock_steps, 6), device=device) * 5.0 
        
        test_inputs = torch.cat([visible_inputs, blind_inputs, shock_inputs], dim=1)
        
        _, _, semantic_projs = model(test_inputs)
        
        shock_idx = half_idx + blind_steps
        
        # Measure Semantic Energy before the shock
        pre_shock_proj = semantic_projs[:, shock_idx - 1, :]
        # Measure Semantic Energy after the shock has processed for a few steps
        post_shock_proj = semantic_projs[:, shock_idx + 5, :] 
        
        energy_change = torch.mean((post_shock_proj - pre_shock_proj)**2).item()
        
        print(f"Semantic Energy Change during Teleport Shock: {energy_change:.6f}")
        
        if energy_change > 0.05:
            print("✅ PASSED (PLASTICITY): The attractor successfully broke its rigid state and adapted to the contradictory dynamics!")
        else:
            print("❌ FAILED (RIGIDITY): The model ignored the shock. It is over-regularized.")

if __name__ == "__main__":
    plasticity_test()
