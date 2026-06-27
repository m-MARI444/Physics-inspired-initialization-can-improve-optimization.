import sys
import torch
import torch.nn as nn
import numpy as np

# Add Training_AI to python path
sys.path.append("/home/goatrobotics/Training_AI")

from pssa.models import MLP, clone_model
from pssa.datasets import get_dataloaders
from pssa.optimizer import optimize_least_action
from run_experiments import init_xavier, init_orthogonal

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between representations X and Y.
    Value is in [0, 1], where 1 indicates identical representation up to orthogonal rotation/scaling.
    X: tensor of shape (B, D1)
    Y: tensor of shape (B, D2)
    """
    # Center the matrices
    X_centered = X - torch.mean(X, dim=0, keepdim=True)
    Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)
    
    # Compute cross-covariance Frobenius norm squared
    cov = torch.matmul(X_centered.t(), Y_centered)
    num = torch.sum(cov ** 2)
    
    # Compute self-covariance Frobenius norms
    cov_xx = torch.matmul(X_centered.t(), X_centered)
    cov_yy = torch.matmul(Y_centered.t(), Y_centered)
    den = torch.sqrt(torch.sum(cov_xx ** 2) * torch.sum(cov_yy ** 2) + 1e-8)
    
    return (num / den).item()

def main():
    print("=== Computing Representational Similarity (Linear CKA) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load MNIST subset for speed
    train_loader, test_loader = get_dataloaders('mnist', batch_size=256)
    
    # Get a validation batch
    for x_val, _ in test_loader:
        x_val = x_val.to(device)
        break
        
    # 2. Setup Architectures
    # Teacher: 784 -> 256 -> 128 -> 10
    # Student: 784 -> 128 -> 64 -> 10
    teacher = MLP(784, [256, 128], 10).to(device)
    student_base = MLP(784, [128, 64], 10).to(device)
    
    # Train teacher for 3 epochs (gives a partially trained teacher representation for comparison)
    print("Pre-training teacher model for 3 epochs...")
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    
    teacher.train()
    for epoch in range(3):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            teacher_opt.zero_grad()
            out, _ = teacher(x)
            loss = criterion(out, y)
            loss.backward()
            teacher_opt.step()
            
    teacher.eval()
    with torch.no_grad():
        _, t_acts = teacher(x_val)
    
    # We will compare Student Layer 1 (width 128) vs Teacher Layer 2 (width 128)
    # We will also compare Student Logits (width 10) vs Teacher Logits (width 10)
    
    runs = [
        {"name": "Xavier Normal", "init_fn": init_xavier, "is_pssa": False},
        {"name": "Orthogonal", "init_fn": init_orthogonal, "is_pssa": False},
        {
            "name": "Distillation-only", 
            "is_pssa": True,
            "config": {
                "K": 1, "m": 0.0, "dt": 1.0, "lr": 5e-3, "num_steps": 50, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.0, "lambda_stab": 0.0, "lambda_info": 0.0
            }
        },
        {
            "name": "PSSA (K=1)", 
            "is_pssa": True,
            "config": {
                "K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 50, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        {
            "name": "PSSA (K=3)", 
            "is_pssa": True,
            "config": {
                "K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 50, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        }
    ]
    
    print("\nCKA Representation Similarity results at Initialization:")
    print(f"{'Initialization Method':<22} | {'Layer 1 Alignment':<18} | {'Logits Alignment':<18}")
    print("-" * 66)
    
    for run in runs:
        student = clone_model(student_base)
        
        if run["is_pssa"]:
            init_xavier(student)
            student, _ = optimize_least_action(student, teacher, train_loader, run["config"])
        else:
            run["init_fn"](student)
            
        student.eval()
        with torch.no_grad():
            _, s_acts = student(x_val)
            
        # Student Layer 1 (index 1) vs Teacher Layer 2 (index 2)
        cka_layer = linear_cka(s_acts[1], t_acts[2])
        # Student Logits (index 3) vs Teacher Logits (index 3)
        cka_logits = linear_cka(s_acts[3], t_acts[3])
        
        print(f"{run['name']:<22} | {cka_layer:.6f}           | {cka_logits:.6f}")

if __name__ == "__main__":
    main()
