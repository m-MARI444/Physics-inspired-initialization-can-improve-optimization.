import torch
import torch.nn as nn
import torch.optim as optim
import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator

def run_action_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 6: Action Probe Benchmark on {device}")
    
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
    
    # 1. Extract frozen latent states
    with torch.no_grad():
        _, _, states, _, _, _, _ = model(norm_trajectory, norm_actions, return_kl=True, return_stats=True, return_sigma=True)
        # states: [1, seq, 64]
    
    # 2. Prepare Probe Dataset
    # We want to predict the action a_t from the latent state s_{t+1} (or transition s_t -> s_{t+1})
    # Since D(s_t, a_t) = s_{t+1}, we can probe if a_t is recoverable from s_{t+1}
    X = states[0, 1:, :] # [seq-1, 64]
    Y_raw = actions[0, :-1, :] # raw unnormalized actions [seq-1, 6]
    
    # Discretize actions into classes for probing
    # 0: Stationary, 1: Base Left, 2: Base Right, 3: Arm Up, 4: Arm Down
    Y_class = torch.zeros(Y_raw.shape[0], dtype=torch.long)
    for i in range(Y_raw.shape[0]):
        a = Y_raw[i]
        if torch.norm(a) < 0.01: Y_class[i] = 0
        elif a[0] > 0.02: Y_class[i] = 1
        elif a[0] < -0.02: Y_class[i] = 2
        elif a[1] > 0.02: Y_class[i] = 3
        elif a[1] < -0.02: Y_class[i] = 4
        else: Y_class[i] = 0
    
    # 3. Train Linear Probe
    probe = nn.Linear(64, 5).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=0.01)
    ce_loss = nn.CrossEntropyLoss()
    
    print("Training linear probe on frozen latents...")
    for epoch in range(100):
        optimizer.zero_grad()
        preds = probe(X)
        loss = ce_loss(preds, Y_class)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    with torch.no_grad():
        final_preds = torch.argmax(probe(X), dim=1)
        accuracy = (final_preds == Y_class).float().mean().item()
        
    print(f"\n--- Action Probe Results ---")
    print(f"Probe Accuracy: {accuracy*100:.2f}%")
    if accuracy > 0.8:
        print("✅ PASSED: The latent states are semantically correct (predicting the true causal actions).")
    else:
        print("❌ FAILED: The latent states diverged, but lost semantic meaning.")

if __name__ == "__main__":
    run_action_probe()
