import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from pssa.models import MLP, clone_model
from pssa.datasets import get_dataloaders, SyntheticDataset
from pssa.optimizer import optimize_least_action
from pssa.utils import (
    measure_layer_variances,
    measure_singular_values,
    evaluate_model,
    train_model
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def init_orthogonal(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def get_effective_rank(singular_values_dict):
    ranks = {}
    for name, s_list in singular_values_dict.items():
        if len(s_list) > 1:
            s = np.array(s_list)
            sum_s = np.sum(s)
            if sum_s > 0:
                p = s / sum_s
                entropy = -np.sum(p * np.log(p + 1e-12))
                ranks[name] = float(entropy)
            else:
                ranks[name] = 0.0
        else:
            ranks[name] = 0.0
    return ranks

def estimate_mlp_fwd_flops(mlp, batch_size):
    flops = 0
    for layer in mlp.layers:
        flops += batch_size * 2 * layer.in_features * layer.out_features
    flops += batch_size * 2 * mlp.out_layer.in_features * mlp.out_layer.out_features
    return flops

def main():
    parser = argparse.ArgumentParser(description="PSSA Refined Weight Initialization Experiments")
    parser.add_argument("--task", type=str, default="spiral", choices=["spiral", "mnist"],
                        help="Task dataset to run experiments on")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train student models")
    parser.add_argument("--teacher_epochs", type=int, default=25,
                        help="Number of epochs to train the teacher model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (cpu, cuda, or auto)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="List of random seeds for replication")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples for synthetic spiral task")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Running refined experiments on: {device} with seeds {args.seeds}")
    
    # 1. Setup Datasets & Disjoint Optimization Loaders to prevent data leakage
    if args.task == "spiral":
        input_dim = 2
        output_dim = 3
        batch_size = 128
        
        # Primary train/test loaders
        # Using a fixed seed for the dataset split itself to ensure same evaluation domain
        train_loader, test_loader = get_dataloaders(
            'spiral', batch_size=batch_size, n_samples=args.num_samples, n_classes=output_dim, seed=42
        )
        
        # Disjoint Optimization Loader for initialization (zero overlap to prevent data leakage)
        optim_loader, _ = get_dataloaders(
            'spiral', batch_size=batch_size, n_samples=1000, n_classes=output_dim, seed=999
        )
        
        teacher = MLP(input_dim, [128, 128], output_dim).to(device)
        student_base = MLP(input_dim, [64, 64], output_dim).to(device)
        
    elif args.task == "mnist":
        input_dim = 784
        output_dim = 10
        batch_size = 256
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full MNIST training dataset
        mnist_full_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        # Split into disjoint training and optimization sets (55k training, 5k optimization)
        # This guarantees zero overlap and prevents data leakage in physics-only activations
        train_subset, optim_subset = random_split(
            mnist_full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        optim_loader = DataLoader(optim_subset, batch_size=batch_size, shuffle=True)
        
        mnist_test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        
        teacher = MLP(input_dim, [256, 128], output_dim).to(device)
        student_base = MLP(input_dim, [128, 64], output_dim).to(device)
        
    criterion = nn.CrossEntropyLoss()
    
    # 2. Train Teacher Model (once on first seed, evaluating on test set)
    print(f"\n=== Training Teacher Model ({args.task}) ===")
    set_seed(args.seeds[0])
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    teacher_history = train_model(
        teacher, train_loader, test_loader, criterion, teacher_optimizer, args.teacher_epochs, device
    )
    final_teacher_loss, final_teacher_acc = evaluate_model(teacher, test_loader, criterion, device)
    print(f"Teacher Model trained successfully. Final Test Accuracy: {final_teacher_acc:.4f}")
    
    results = {
        "task": args.task,
        "teacher_accuracy": final_teacher_acc,
        "runs": {}
    }
    
    # Define experimental configurations (Ablation Conditions & K Path sweeps)
    runs_to_execute = [
        {
            "name": "Xavier Normal",
            "type": "baseline",
            "init_fn": init_xavier
        },
        {
            "name": "Orthogonal",
            "type": "baseline",
            "init_fn": init_orthogonal
        },
        {
            "name": "Distillation-only",
            "type": "pssa_variant",
            "config": {
                "K": 1, "m": 0.0, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.0, "lambda_stab": 0.0, "lambda_info": 0.0
            }
        },
        # Condition 3a: E_conn only (Truly data-free weight-space properties)
        {
            "name": "Condition 3a (E_conn, K=1)",
            "type": "pssa",
            "config": {
                "K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 0.0, "lambda_info": 0.0,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        {
            "name": "Condition 3a (E_conn, K=3)",
            "type": "pssa",
            "config": {
                "K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 0.0, "lambda_info": 0.0,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        # Condition 3b: E_stab + E_info only (Activation-dependent constraints)
        {
            "name": "Condition 3b (E_stab+E_info, K=1)",
            "type": "pssa",
            "config": {
                "K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.0, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        {
            "name": "Condition 3b (E_stab+E_info, K=3)",
            "type": "pssa",
            "config": {
                "K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.0, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        # Condition 3c: All three physics (E_conn + E_stab + E_info)
        {
            "name": "Condition 3c (All Physics, K=1)",
            "type": "pssa",
            "config": {
                "K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        {
            "name": "Condition 3c (All Physics, K=3)",
            "type": "pssa",
            "config": {
                "K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        # Condition 4: Guided PSSA (E_pred + all physics)
        {
            "name": "Condition 4 (Guided PSSA, K=1)",
            "type": "pssa",
            "config": {
                "K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        },
        {
            "name": "Condition 4 (Guided PSSA, K=3)",
            "type": "pssa",
            "config": {
                "K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device,
                "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5,
                "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0
            }
        }
    ]
    
    # Pre-calculate layer-wise FLOP sizes for compute efficiency tracking
    fwd_flops_teacher = estimate_mlp_fwd_flops(teacher, batch_size)
    fwd_flops_student = estimate_mlp_fwd_flops(student_base, batch_size)
    bwd_flops_student = 2 * fwd_flops_student  # Standard backpropagation approximation
    
    # Calculate training FLOPs per epoch
    num_batches = len(train_loader)
    train_flops_per_epoch = num_batches * (fwd_flops_student + bwd_flops_student)
    
    for run in runs_to_execute:
        name = run["name"]
        print(f"\n--- Running: {name} ---")
        
        # Lists to aggregate across seeds
        seed_results = {
            "test_acc_history": [],
            "test_loss_history": [],
            "train_acc_history": [],
            "train_loss_history": [],
            "init_time": [],
            "init_flops": [],
            "step0_ranks": [],
            "step0_variances": []
        }
        
        for seed in args.seeds:
            print(f"  > Seed {seed}...")
            set_seed(seed)
            
            # Clone student base structure
            student = clone_model(student_base)
            
            init_time = 0.0
            init_flops = 0.0
            
            if run["type"] == "baseline":
                # Apply baseline weight init
                start_t = time.time()
                run["init_fn"](student)
                init_time = time.time() - start_t
                init_flops = 0.0
            else:
                # PSSA optimization starting from Xavier initial state
                init_xavier(student)
                
                # Solve Stationary Action
                opt_config = run["config"]
                
                # Track wall-clock time of initialization
                start_t = time.time()
                # Run optimization on the disjoint optim_loader (no training data overlap)
                student, _ = optimize_least_action(student, teacher, optim_loader, opt_config)
                init_time = time.time() - start_t
                
                # Calculate estimated FLOPs for the path optimization
                K_val = opt_config["K"]
                num_steps = opt_config["num_steps"]
                # For each step: Teacher Fwd + K * (Student Fwd + Student Bwd)
                init_flops = num_steps * (fwd_flops_teacher + K_val * (fwd_flops_student + bwd_flops_student))
            
            # Step-0 measurements (Before any gradient updates/fine-tuning)
            init_variances = measure_layer_variances(student, train_loader, device)
            init_singular_values = measure_singular_values(student)
            init_ranks = get_effective_rank(init_singular_values)
            
            # Fine-tune the student model
            student_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
            student_history = train_model(
                student, train_loader, test_loader, criterion, student_optimizer, args.epochs, device
            )
            
            final_loss, final_acc = evaluate_model(student, test_loader, criterion, device)
            print(f"    Finished seed {seed}. Test Accuracy: {final_acc:.4f} | Init Time: {init_time:.2f}s | Init FLOPs: {init_flops:.2e}")
            
            # Accumulate metrics
            seed_results["test_acc_history"].append(student_history["test_acc"])
            seed_results["test_loss_history"].append(student_history["test_loss"])
            seed_results["train_acc_history"].append(student_history["train_loss"]) # standard uses train_loss
            seed_results["train_loss_history"].append(student_history["train_loss"])
            seed_results["init_time"].append(init_time)
            seed_results["init_flops"].append(init_flops)
            seed_results["step0_ranks"].append(init_ranks)
            seed_results["step0_variances"].append(init_variances)
            
        # Log aggregated results across seeds
        results["runs"][name] = {
            "type": run["type"],
            "test_acc_seeds": seed_results["test_acc_history"],
            "test_loss_seeds": seed_results["test_loss_history"],
            "init_time_seeds": seed_results["init_time"],
            "init_flops_seeds": seed_results["init_flops"],
            "step0_ranks_seeds": seed_results["step0_ranks"],
            "step0_variances_seeds": seed_results["step0_variances"],
            "train_flops_per_epoch": train_flops_per_epoch
        }
        
    # Save results to JSON
    output_filename = f"results_{args.task}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll experiments completed! Results saved to {output_filename}")

if __name__ == "__main__":
    main()
