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
from huggingface_hub import hf_hub_download, HfApi

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

def upload_to_hf(local_file, path_in_repo, repo_id, token):
    if not repo_id or not token:
        return
    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"📡 Uploaded '{local_file}' to Hugging Face repository '{repo_id}' at '{path_in_repo}'")
    except Exception as e:
        print(f"Warning: Failed to upload '{local_file}' to Hugging Face: {e}")

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
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save output files")
    parser.add_argument("--hf_repo", type=str, default="",
                        help="Hugging Face repository ID to persist checkpoints")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face API write token")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Running refined experiments on: {device} with seeds {args.seeds}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"results_{args.task}.json")
    teacher_checkpoint = f"checkpoints/teacher_{args.task}.pth"
    
    results = {"task": args.task, "teacher_accuracy": 0.0, "runs": {}}
    
    # 1. Recover from Hugging Face if config matches
    if args.hf_repo and args.hf_token:
        print(f"Checking Hugging Face repository '{args.hf_repo}' for existing checkpoints...")
        try:
            api = HfApi(token=args.hf_token)
            files = api.list_repo_files(repo_id=args.hf_repo)
            
            # Download results file
            results_name = f"results_{args.task}.json"
            if results_name in files:
                hf_hub_download(repo_id=args.hf_repo, filename=results_name, token=args.hf_token, local_dir=".")
                print(f"Successfully recovered '{results_name}' from Hugging Face.")
                
            # Download teacher file
            teacher_name = f"checkpoints/teacher_{args.task}.pth"
            if teacher_name in files:
                os.makedirs("checkpoints", exist_ok=True)
                hf_hub_download(repo_id=args.hf_repo, filename=teacher_name, token=args.hf_token, local_dir=".")
                print(f"Successfully recovered teacher checkpoint from Hugging Face.")
        except Exception as e:
            print(f"Note: Could not recover from HF Hub: {e}")

    # Check local filesystem
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                loaded_results = json.load(f)
                if loaded_results.get("task") == args.task:
                    results = loaded_results
                    print(f"Resuming from results file: {len(results.get('runs', {}))} configurations already completed.")
        except Exception as e:
            print(f"Warning: Could not load existing results file: {e}")

    os.makedirs("checkpoints", exist_ok=True)

    # Setup Datasets & Disjoint Optimization Loaders to prevent data leakage
    if args.task == "spiral":
        input_dim = 2
        output_dim = 3
        batch_size = 128
        
        train_loader, test_loader = get_dataloaders(
            'spiral', batch_size=batch_size, n_samples=args.num_samples, n_classes=output_dim, seed=42
        )
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
        
        mnist_full_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
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
    
    # Train or Load Teacher Model
    print(f"\n=== Setting up Teacher Model ({args.task}) ===")
    if os.path.exists(teacher_checkpoint) and results.get("teacher_accuracy", 0.0) > 0.0:
        try:
            teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
            print(f"Loaded pre-trained Teacher model from checkpoint. Accuracy: {results['teacher_accuracy']:.4f}")
        except Exception as e:
            print(f"Failed to load teacher checkpoint ({e}). Re-training...")
            results["teacher_accuracy"] = 0.0
            
    if results.get("teacher_accuracy", 0.0) == 0.0:
        print("Training Teacher Model...")
        set_seed(args.seeds[0])
        teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
        _ = train_model(
            teacher, train_loader, test_loader, criterion, teacher_optimizer, args.teacher_epochs, device
        )
        final_teacher_loss, final_teacher_acc = evaluate_model(teacher, test_loader, criterion, device)
        results["teacher_accuracy"] = final_teacher_acc
        torch.save(teacher.state_dict(), teacher_checkpoint)
        print(f"Teacher Model trained successfully. Final Test Accuracy: {final_teacher_acc:.4f}")
        
        # Save and upload teacher immediately to HF
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        upload_to_hf(output_filename, f"results_{args.task}.json", args.hf_repo, args.hf_token)
        upload_to_hf(teacher_checkpoint, f"checkpoints/teacher_{args.task}.pth", args.hf_repo, args.hf_token)
            
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
    bwd_flops_student = 2 * fwd_flops_student
    num_batches = len(train_loader)
    train_flops_per_epoch = num_batches * (fwd_flops_student + bwd_flops_student)
    
    for run in runs_to_execute:
        name = run["name"]
        
        # Check if already done (allows resuming safely)
        if name in results.get("runs", {}):
            print(f"Skipping completed run: {name}")
            continue
            
        print(f"\n--- Running: {name} ---")
        
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
            
            student = clone_model(student_base)
            init_time = 0.0
            init_flops = 0.0
            
            if run["type"] == "baseline":
                start_t = time.time()
                run["init_fn"](student)
                init_time = time.time() - start_t
                init_flops = 0.0
            else:
                init_xavier(student)
                opt_config = run["config"]
                
                start_t = time.time()
                student, _ = optimize_least_action(student, teacher, optim_loader, opt_config)
                init_time = time.time() - start_t
                
                K_val = opt_config["K"]
                num_steps = opt_config["num_steps"]
                init_flops = num_steps * (fwd_flops_teacher + K_val * (fwd_flops_student + bwd_flops_student))
            
            init_variances = measure_layer_variances(student, train_loader, device)
            init_singular_values = measure_singular_values(student)
            init_ranks = get_effective_rank(init_singular_values)
            
            student_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
            student_history = train_model(
                student, train_loader, test_loader, criterion, student_optimizer, args.epochs, device
            )
            
            final_loss, final_acc = evaluate_model(student, test_loader, criterion, device)
            print(f"    Finished seed {seed}. Test Accuracy: {final_acc:.4f} | Init Time: {init_time:.2f}s | Init FLOPs: {init_flops:.2e}")
            
            seed_results["test_acc_history"].append(student_history["test_acc"])
            seed_results["test_loss_history"].append(student_history["test_loss"])
            seed_results["train_acc_history"].append(student_history["train_loss"])
            seed_results["train_loss_history"].append(student_history["train_loss"])
            seed_results["init_time"].append(init_time)
            seed_results["init_flops"].append(init_flops)
            seed_results["step0_ranks"].append(init_ranks)
            seed_results["step0_variances"].append(init_variances)
            
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
        
        # Save checkpoint locally
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved checkpoint results for '{name}' to {output_filename}")
        
        # Upload checkpoint to HF
        upload_to_hf(output_filename, f"results_{args.task}.json", args.hf_repo, args.hf_token)
        
    print(f"\nAll experiments completed! Final results saved to {output_filename}")

if __name__ == "__main__":
    main()
