import argparse
import json
import os
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ==========================================
# 1. Model and Dataset Definition
# ==========================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.out_layer = nn.Linear(in_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        activations = [x]
        curr = x
        for layer in self.layers:
            curr = self.activation_fn(layer(curr))
            activations.append(curr)
        out = self.out_layer(curr)
        activations.append(out)
        return out, activations

def clone_model(model):
    return copy.deepcopy(model)

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_spiral_data(n_samples=1500, n_classes=3, noise=0.2, seed=42):
    np.random.seed(seed)
    N = n_samples // n_classes
    D = 2
    X = np.zeros((N * n_classes, D))
    y = np.zeros(N * n_classes, dtype='int64')
    for j in range(n_classes):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def get_dataloaders(task_name='spiral', batch_size=128, n_samples=1500, n_classes=3, noise=0.2, seed=42):
    if task_name == 'spiral':
        X, y = get_spiral_data(n_samples=n_samples, n_classes=n_classes, noise=noise, seed=seed)
        indices = torch.randperm(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]
        train_dataset = SyntheticDataset(X[train_idx], y[train_idx])
        test_dataset = SyntheticDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    elif task_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown task name: {task_name}")

# ==========================================
# 2. Physics-Based Energy Formulations
# ==========================================

def compute_prediction_energy(student_activations, teacher_activations, align_layers=None):
    if align_layers is None:
        s_out = student_activations[-1]
        t_out = teacher_activations[-1]
        return torch.mean((s_out - t_out) ** 2)
    loss = 0.0
    count = 0
    for idx in align_layers:
        if idx < len(student_activations) and idx < len(teacher_activations):
            s_act = student_activations[idx]
            t_act = teacher_activations[idx]
            if s_act.shape == t_act.shape:
                loss += torch.mean((s_act - t_act) ** 2)
                count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=student_activations[-1].device)

def compute_connection_energy(model, eta=1e-4, beta=1e-3):
    loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            w = param.view(param.size(0), -1)
            loss += 0.5 * eta * torch.sum(w ** 2)
            if w.size(0) > 1:
                row_norms = torch.norm(w, dim=1, keepdim=True) + 1e-8
                w_norm = w / row_norms
                corr = torch.matmul(w_norm, w_norm.t())
                eye = torch.eye(corr.size(0), device=corr.device)
                loss += 0.5 * beta * torch.mean((corr - eye) ** 2)
    return loss

def compute_stability_energy(student_activations, alpha=1.0):
    loss = 0.0
    variances = []
    for act in student_activations[1:-1]:
        act_flat = act.view(act.size(0), -1)
        feature_vars = torch.var(act_flat, dim=0)
        variances.append(torch.mean(feature_vars))
    if not variances:
        return torch.tensor(0.0, device=student_activations[-1].device)
    for i in range(1, len(variances)):
        loss += (variances[i] - variances[i-1]) ** 2
    for var in variances:
        loss += (var - 1.0) ** 2
    return alpha * loss

def compute_information_energy(student_activations, mu=1.0):
    loss = 0.0
    count = 0
    for act in student_activations[1:-1]:
        act_flat = act.view(act.size(0), -1)
        batch_size, num_features = act_flat.shape
        if batch_size <= 1 or num_features <= 1:
            continue
        act_centered = act_flat - torch.mean(act_flat, dim=0, keepdim=True)
        cov = torch.matmul(act_centered.t(), act_centered) / (batch_size - 1 + 1e-8)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(0))
        eye = torch.eye(num_features, device=corr.device)
        loss += torch.mean((corr - eye) ** 2)
        count += 1
    return mu * (loss / count) if count > 0 else torch.tensor(0.0, device=student_activations[-1].device)

def compute_potential_energy(student_model, student_activations, teacher_activations, config):
    v_pred = compute_prediction_energy(student_activations, teacher_activations, align_layers=config.get('align_layers', None))
    v_conn = compute_connection_energy(student_model, eta=config.get('eta', 1e-4), beta=config.get('beta', 1e-3))
    v_stab = compute_stability_energy(student_activations, alpha=config.get('alpha', 1.0))
    v_info = compute_information_energy(student_activations, mu=config.get('mu', 1.0))
    total_potential = (
        config.get('lambda_pred', 1.0) * v_pred +
        config.get('lambda_conn', 1.0) * v_conn +
        config.get('lambda_stab', 1.0) * v_stab +
        config.get('lambda_info', 1.0) * v_info
    )
    breakdown = {
        "potential_total": total_potential.item(),
        "v_pred": v_pred.item(),
        "v_conn": v_conn.item(),
        "v_stab": v_stab.item(),
        "v_info": v_info.item()
    }
    return total_potential, breakdown

# ==========================================
# 3. Least-Action Path Optimization
# ==========================================

def compute_kinetic_energy(model_curr, model_prev, m=1.0, dt=1.0):
    kinetic = 0.0
    for p_curr, p_prev in zip(model_curr.parameters(), model_prev.parameters()):
        if p_curr.requires_grad:
            kinetic += torch.sum((p_curr - p_prev) ** 2)
    return 0.5 * m * kinetic / (dt ** 2)

def optimize_least_action(base_model, teacher_model, data_loader, config):
    device = config.get('device', 'cpu')
    K = config.get('K', 3)
    m = config.get('m', 1.0)
    dt = config.get('dt', 1.0)
    lr = config.get('lr', 1e-3)
    num_steps = config.get('num_steps', 150)
    
    base_model = base_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    path = []
    w0 = clone_model(base_model)
    for p in w0.parameters():
        p.requires_grad = False
    path.append(w0)
    
    params_to_optimize = []
    for k in range(1, K + 1):
        wk = clone_model(base_model)
        for p in wk.parameters():
            p.requires_grad = True
            params_to_optimize.append(p)
        path.append(wk)
        
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    data_iter = iter(data_loader)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, _ = next(data_iter)
        x = x.to(device)
        
        with torch.no_grad():
            t_out, t_acts = teacher_model(x)
            
        total_action = torch.tensor(0.0, device=device)
        for k in range(1, K + 1):
            t_val = compute_kinetic_energy(path[k], path[k-1], m=m, dt=dt)
            s_out_k, s_acts_k = path[k](x)
            v_val, _ = compute_potential_energy(path[k], s_acts_k, t_acts, config)
            total_action = total_action + t_val + dt * v_val
            
        total_action.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
        optimizer.step()
        
    opt_student = clone_model(path[-1])
    return opt_student

# ==========================================
# 4. Utilities
# ==========================================

def measure_layer_variances(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, activations = model(x)
            variances = []
            for act in activations:
                act_flat = act.view(act.size(0), -1)
                feature_vars = torch.var(act_flat, dim=0)
                variances.append(torch.mean(feature_vars).item())
            return variances

def measure_singular_values(model):
    singular_vals = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            w = param.view(param.size(0), -1).detach().cpu()
            if w.size(0) > 1 and w.size(1) > 1:
                try:
                    _, s, _ = torch.svd(w)
                    singular_vals[name] = s.tolist()
                except Exception:
                    singular_vals[name] = []
            else:
                singular_vals[name] = []
    return singular_vals

def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return total_loss / total, correct / total

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu'):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
    return history

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

# ==========================================
# 5. Core Experiments Orchestration
# ==========================================

def run_experiments(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running experiments on: {device} | Seeds: {args.seeds}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"results_{args.task}.json")
    
    # Check for existing results to enable resume capability
    results = {"task": args.task, "teacher_accuracy": 0.0, "runs": {}}
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                loaded_results = json.load(f)
                if loaded_results.get("task") == args.task:
                    results = loaded_results
                    print(f"Loaded existing results file '{output_filename}'. Resuming from previous run...")
        except Exception as e:
            print(f"Warning: Could not load existing results file: {e}. Starting fresh.")

    # 1. Setup Datasets & Disjoint Optimization Loaders to prevent data leakage
    if args.task == "spiral":
        input_dim = 2
        output_dim = 3
        batch_size = 128
        train_loader, test_loader = get_dataloaders('spiral', batch_size=batch_size, n_samples=args.num_samples, n_classes=output_dim, seed=42)
        optim_loader, _ = get_dataloaders('spiral', batch_size=batch_size, n_samples=1000, n_classes=output_dim, seed=999)
        teacher = MLP(input_dim, [128, 128], output_dim).to(device)
        student_base = MLP(input_dim, [64, 64], output_dim).to(device)
    elif args.task == "mnist":
        input_dim = 784
        output_dim = 10
        batch_size = 256
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_subset, optim_subset = random_split(mnist_full_train, [55000, 5000], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        optim_loader = DataLoader(optim_subset, batch_size=batch_size, shuffle=True)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        teacher = MLP(input_dim, [256, 128], output_dim).to(device)
        student_base = MLP(input_dim, [128, 64], output_dim).to(device)
        
    criterion = nn.CrossEntropyLoss()
    
    # 2. Train or Load Teacher Model
    print(f"\n=== Setting up Teacher Model ({args.task}) ===")
    os.makedirs("checkpoints", exist_ok=True)
    teacher_checkpoint = f"checkpoints/teacher_{args.task}.pth"
    
    if os.path.exists(teacher_checkpoint) and results.get("teacher_accuracy", 0.0) > 0.0:
        try:
            teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
            print(f"Loaded pre-trained Teacher model from checkpoint. Accuracy: {results['teacher_accuracy']:.4f}")
        except Exception as e:
            print(f"Failed to load teacher checkpoint ({e}). Re-training...")
            results["teacher_accuracy"] = 0.0

    if results.get("teacher_accuracy", 0.0) == 0.0:
        print("Training Teacher Model...")
        random.seed(args.seeds[0])
        np.random.seed(args.seeds[0])
        torch.manual_seed(args.seeds[0])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seeds[0])
        teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
        _ = train_model(teacher, train_loader, test_loader, criterion, teacher_optimizer, args.teacher_epochs, device)
        _, final_teacher_acc = evaluate_model(teacher, test_loader, criterion, device)
        results["teacher_accuracy"] = final_teacher_acc
        torch.save(teacher.state_dict(), teacher_checkpoint)
        print(f"Teacher Model trained successfully. Final Test Accuracy: {final_teacher_acc:.4f}")
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
            
    runs_to_execute = [
        {"name": "Xavier Normal", "type": "baseline", "init_fn": lambda m: nn.init.xavier_normal_(m.weight)},
        {"name": "Orthogonal", "type": "baseline", "init_fn": lambda m: nn.init.orthogonal_(m.weight)},
        {"name": "Distillation-only", "type": "pssa_variant", "config": {"K": 1, "m": 0.0, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 1.0, "lambda_conn": 0.0, "lambda_stab": 0.0, "lambda_info": 0.0}},
        {"name": "Condition 3a (E_conn, K=1)", "type": "pssa", "config": {"K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 0.0, "lambda_info": 0.0, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 3a (E_conn, K=3)", "type": "pssa", "config": {"K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 0.0, "lambda_info": 0.0, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 3b (E_stab+E_info, K=1)", "type": "pssa", "config": {"K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.0, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 3b (E_stab+E_info, K=3)", "type": "pssa", "config": {"K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.0, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 3c (All Physics, K=1)", "type": "pssa", "config": {"K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 3c (All Physics, K=3)", "type": "pssa", "config": {"K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 0.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 4 (Guided PSSA, K=1)", "type": "pssa", "config": {"K": 1, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}},
        {"name": "Condition 4 (Guided PSSA, K=3)", "type": "pssa", "config": {"K": 3, "m": 0.5, "dt": 1.0, "lr": 5e-3, "num_steps": 150, "device": device, "lambda_pred": 1.0, "lambda_conn": 0.2, "lambda_stab": 1.0, "lambda_info": 0.5, "eta": 1e-4, "beta": 1e-3, "alpha": 1.0, "mu": 1.0}}
    ]
    
    fwd_flops_teacher = estimate_mlp_fwd_flops(teacher, batch_size)
    fwd_flops_student = estimate_mlp_fwd_flops(student_base, batch_size)
    bwd_flops_student = 2 * fwd_flops_student
    train_flops_per_epoch = len(train_loader) * (fwd_flops_student + bwd_flops_student)
    
    for run in runs_to_execute:
        name = run["name"]
        
        # Check if already completed and skip
        if name in results.get("runs", {}):
            print(f"Skipping completed run: {name}")
            continue
            
        print(f"\n--- Running: {name} ---")
        
        seed_results = {
            "test_acc_history": [], "test_loss_history": [],
            "init_time": [], "init_flops": [],
            "step0_ranks": [], "step0_variances": []
        }
        
        for seed in args.seeds:
            print(f"  > Seed {seed}...")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            student = clone_model(student_base)
            init_time = 0.0
            init_flops = 0.0
            
            if run["type"] == "baseline":
                start_t = time.time()
                for m in student.modules():
                    if isinstance(m, nn.Linear):
                        run["init_fn"](m)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                init_time = time.time() - start_t
                init_flops = 0.0
            else:
                for m in student.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                opt_config = run["config"]
                start_t = time.time()
                student = optimize_least_action(student, teacher, optim_loader, opt_config)
                init_time = time.time() - start_t
                K_val = opt_config["K"]
                num_steps = opt_config["num_steps"]
                init_flops = num_steps * (fwd_flops_teacher + K_val * (fwd_flops_student + bwd_flops_student))
            
            init_variances = measure_layer_variances(student, train_loader, device)
            init_singular_values = measure_singular_values(student)
            init_ranks = get_effective_rank(init_singular_values)
            
            student_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
            student_history = train_model(student, train_loader, test_loader, criterion, student_optimizer, args.epochs, device)
            final_loss, final_acc = evaluate_model(student, test_loader, criterion, device)
            print(f"    Finished seed {seed}. Test Accuracy: {final_acc:.4f} | Init Time: {init_time:.2f}s | Init FLOPs: {init_flops:.2e}")
            
            seed_results["test_acc_history"].append(student_history["test_acc"])
            seed_results["test_loss_history"].append(student_history["test_loss"])
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
        
        # Save checkpoint to disk after each configuration finishes
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved checkpoint results for '{name}' to {output_filename}")
        
    print(f"\nAll experiments completed! Final results saved to {output_filename}")
    generate_publication_plots(results, args)

# ==========================================
# 6. Publication-Grade Plotting
# ==========================================

def generate_publication_plots(results, args):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 16,
        'legend.fontsize': 9, 'grid.alpha': 0.3
    })
    
    colors = {
        "Xavier Normal": "#7F8C8D", "Orthogonal": "#E67E22", "Distillation-only": "#C0392B",
        "Condition 3a (E_conn, K=1)": "#9B59B6", "Condition 3a (E_conn, K=3)": "#8E44AD",
        "Condition 3b (E_stab+E_info, K=1)": "#16A085", "Condition 3b (E_stab+E_info, K=3)": "#27AE60",
        "Condition 3c (All Physics, K=1)": "#2980B9", "Condition 3c (All Physics, K=3)": "#34495E",
        "Condition 4 (Guided PSSA, K=1)": "#F1C40F", "Condition 4 (Guided PSSA, K=3)": "#D35400"
    }
    
    # Plot 1: Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for run_name, run_data in results["runs"].items():
        test_accs = np.array(run_data["test_acc_seeds"])
        epochs = range(1, test_accs.shape[1] + 1)
        mean_acc = np.mean(test_accs, axis=0)
        std_acc = np.std(test_accs, axis=0)
        color = colors.get(run_name, "#9B59B6")
        axes[0].plot(epochs, mean_acc, label=run_name, color=color, linewidth=2.0)
        axes[0].fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)
    axes[0].set_title("Test Accuracy vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    
    for run_name, run_data in results["runs"].items():
        test_losses = np.array(run_data["test_loss_seeds"])
        epochs = range(1, test_losses.shape[1] + 1)
        mean_loss = np.mean(test_losses, axis=0)
        std_loss = np.std(test_losses, axis=0)
        color = colors.get(run_name, "#9B59B6")
        axes[1].plot(epochs, mean_loss, label=run_name, color=color, linewidth=2.0)
        axes[1].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)
    axes[1].set_title("Test Loss vs Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross Entropy Loss")
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"learning_curves_{args.task}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Compute Efficiency
    plt.figure(figsize=(10, 6))
    for run_name, run_data in results["runs"].items():
        test_accs = np.array(run_data["test_acc_seeds"])
        mean_acc = np.mean(test_accs, axis=0)
        mean_init_flops = np.mean(run_data["init_flops_seeds"])
        train_flops_per_epoch = run_data["train_flops_per_epoch"]
        num_epochs = len(mean_acc)
        cumulative_flops = [mean_init_flops + e * train_flops_per_epoch for e in range(1, num_epochs + 1)]
        color = colors.get(run_name, "#9B59B6")
        plt.plot(cumulative_flops, mean_acc, label=run_name, color=color, linewidth=2.0, marker='o', markersize=4)
    plt.title("Compute Efficiency: Test Accuracy vs Cumulative FLOPs")
    plt.xlabel("Total FLOPs (Log Scale)")
    plt.ylabel("Test Accuracy")
    plt.xscale("log")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, f"compute_efficiency_{args.task}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Activation Variance
    plt.figure(figsize=(10, 6))
    for run_name, run_data in results["runs"].items():
        variances_seeds = np.array(run_data["step0_variances_seeds"])
        mean_vars = np.mean(variances_seeds, axis=0)
        layers = range(len(mean_vars))
        color = colors.get(run_name, "#9B59B6")
        plt.plot(layers, mean_vars, label=run_name, color=color, linewidth=2.0, marker='s', markersize=5)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Ideal Stability Target (1.0)")
    plt.title("Step-0 Activation Variance Layer-by-Layer")
    plt.xlabel("Layer Index (0=Input, L=Output)")
    plt.ylabel("Variance")
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.output_dir, f"activation_variances_{args.task}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Effective Rank
    plt.figure(figsize=(12, 6))
    run_names = []
    mean_ranks = []
    std_ranks = []
    for run_name, run_data in results["runs"].items():
        ranks_seeds = run_data["step0_ranks_seeds"]
        layer_keys = [k for k in ranks_seeds[0].keys() if "layers.0.weight" in k or "conv1.weight" in k]
        if not layer_keys:
            layer_keys = [k for k in ranks_seeds[0].keys() if "weight" in k]
        if layer_keys:
            key = layer_keys[0]
            ranks = [seed_ranks[key] for seed_ranks in ranks_seeds]
            run_names.append(run_name)
            mean_ranks.append(np.mean(ranks))
            std_ranks.append(np.std(ranks))
    if run_names:
        x_indices = np.arange(len(run_names))
        bar_colors = [colors.get(name, "#9B59B6") for name in run_names]
        plt.bar(x_indices, mean_ranks, yerr=std_ranks, align='center', alpha=0.8, color=bar_colors, capsize=10, edgecolor='black', linewidth=1.2)
        plt.xticks(x_indices, run_names, rotation=45, ha="right")
        plt.ylabel("Weight Matrix Singular Value Entropy (Effective Rank)")
        plt.title("Step-0 Weight Matrix Effective Rank Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"singular_values_{args.task}.png"), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSSA Refined Kaggle Experiments")
    parser.add_argument("--task", type=str, default="spiral", choices=["spiral", "mnist"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--teacher_epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    run_experiments(args)
