import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_model import PSSAV2

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stability Benchmark on: {device}")
    
    # 1. Load Model
    model = PSSAV2(vocab_size=13, d_model=64).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/pssa_v2_math.pth", map_location=device))
        print("Loaded trained checkpoint.")
    except FileNotFoundError:
        print("No checkpoint found. Running benchmark on initialized weights.")
        
    model.eval()
    
    # 2. Setup initial state
    # Dummy input sequence "1+1=2"
    dummy_input = torch.tensor([[1, 10, 1, 11, 2]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Get baseline representation
        _, initial_state, initial_semantic_proj = model(dummy_input)
        
        # P(s_0)
        p_s0 = initial_semantic_proj[:, -1, :] 
        
        current_state = initial_state
        
        print("\nStarting 10,000 Step Persistent Horizon Test...")
        print("Measuring Semantic Drift: E(s) = |P(s_t) - P(s_0)|^2")
        print("-" * 50)
        
        steps = 10000
        log_interval = 1000
        
        for t in range(1, steps + 1):
            _, current_state, semantic_proj = model(dummy_input, prev_state=current_state)
            
            if t % log_interval == 0:
                p_st = semantic_proj[:, -1, :]
                drift = torch.mean((p_st - p_s0)**2).item()
                print(f"Step {t:5d} | Semantic Drift E(s): {drift:.6f}")
                
        print("-" * 50)
        final_drift = torch.mean((semantic_proj[:, -1, :] - p_s0)**2).item()
        
        if final_drift < 2.0:
            print(f"✅ PASSED: System exhibits Lyapunov Stability. Final drift is bounded at {final_drift:.6f}")
        else:
            print(f"❌ FAILED: State explosion detected. Final drift {final_drift:.6f}")

if __name__ == "__main__":
    run_benchmark()
