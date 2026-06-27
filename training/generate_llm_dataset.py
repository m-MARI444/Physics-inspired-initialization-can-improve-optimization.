import torch
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator

def generate_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating Latent-to-Language Dataset...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    
    traj_path = os.path.join(data_dir, "arm_trajectory.pt")
    act_path  = os.path.join(data_dir, "causal_actions.pt")
    
    trajectory = torch.load(traj_path, map_location=device, weights_only=False)
    actions = torch.load(act_path, map_location=device, weights_only=False)
    
    traj_mean = trajectory.mean(dim=1, keepdim=True)
    traj_std = trajectory.std(dim=1, keepdim=True) + 1e-6
    act_mean = actions.mean(dim=1, keepdim=True)
    act_std = actions.std(dim=1, keepdim=True) + 1e-6
    
    norm_trajectory = (trajectory - traj_mean) / traj_std
    norm_actions = (actions - act_mean) / act_std
    
    model = PSSASimulator(d_in=6, d_action=6, d_model=64, d_out=6, vocab_size=8).to(device)
    model.load_state_dict(torch.load(os.path.join(project_root, "checkpoints", "pssa_v2_simulator.pth"), map_location=device))
    model.eval()
    
    with torch.no_grad():
        _, _, states, _, _, _, _ = model(norm_trajectory, norm_actions, return_kl=True, return_stats=True, return_sigma=True)
        states = states[0] # [seq, 64]
    
    Y_raw = actions[0] # [seq, 6]
    
    descriptions = {
        0: "The robotic arm is currently stationary and holding its physical position.",
        1: "The base of the robotic arm is rotating smoothly to the left.",
        2: "The base of the robotic arm is rotating smoothly to the right.",
        3: "The main arm joint is elevating upwards.",
        4: "The main arm joint is lowering downwards."
    }
    
    dataset = []
    saved_states = []
    
    for i in range(states.shape[0] - 1):
        a = Y_raw[i]
        c = 0
        if torch.norm(a) < 0.01: c = 0
        elif a[0] > 0.02: c = 1
        elif a[0] < -0.02: c = 2
        elif a[1] > 0.02: c = 3
        elif a[1] < -0.02: c = 4
        
        dataset.append({
            "step": i,
            "text": descriptions[c]
        })
        saved_states.append(states[i+1].cpu()) # The state after the action
        
    torch.save(torch.stack(saved_states), os.path.join(data_dir, "llm_latent_states.pt"))
    
    with open(os.path.join(data_dir, "llm_descriptions.json"), "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Generated {len(dataset)} rich language pairs.")

if __name__ == "__main__":
    generate_dataset()
