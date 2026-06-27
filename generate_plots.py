import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="PSSA Refined Plot Generator")
    parser.add_argument("--task", type=str, default="spiral", choices=["spiral", "mnist"],
                        help="Task results to process")
    parser.add_argument("--artifact_dir", type=str, 
                        default="/home/goatrobotics/.gemini/antigravity/brain/6b1090bb-f645-4444-ae7e-98f7c73f876e",
                        help="Directory to save generated plot images")
    args = parser.parse_args()
    
    json_path = f"results_{args.task}.json"
    if not os.path.exists(json_path):
        print(f"Error: Results file '{json_path}' not found! Run the experiments first.")
        return
        
    with open(json_path, "r") as f:
        results = json.load(f)
        
    os.makedirs(args.artifact_dir, exist_ok=True)
        
    # Setup premium plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'legend.fontsize': 9,
        'grid.alpha': 0.3
    })
    
    # Harmonious premium color palette
    colors = {
        "Xavier Normal": "#7F8C8D",                     # Gray
        "Orthogonal": "#E67E22",                        # Orange
        "Distillation-only": "#C0392B",                 # Deep Red
        "Condition 3a (E_conn, K=1)": "#9B59B6",        # Amethyst
        "Condition 3a (E_conn, K=3)": "#8E44AD",        # Wisteria
        "Condition 3b (E_stab+E_info, K=1)": "#16A085",  # Greenish Teal
        "Condition 3b (E_stab+E_info, K=3)": "#27AE60",  # Nephrite Green
        "Condition 3c (All Physics, K=1)": "#2980B9",   # Belize Hole Blue
        "Condition 3c (All Physics, K=3)": "#34495E",   # Wet Asphalt Blue
        "Condition 4 (Guided PSSA, K=1)": "#F1C40F",    # Sun Flower Yellow
        "Condition 4 (Guided PSSA, K=3)": "#D35400"     # Pumpkin Orange
    }
    
    # -------------------------------------------------------------
    # Plot 1: Learning Curves (Train/Test Accuracy and Loss with StdBands)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test Accuracy
    for run_name, run_data in results["runs"].items():
        test_accs = np.array(run_data["test_acc_seeds"]) # [num_seeds, num_epochs]
        epochs = range(1, test_accs.shape[1] + 1)
        mean_acc = np.mean(test_accs, axis=0)
        std_acc = np.std(test_accs, axis=0)
        
        color = colors.get(run_name, "#9B59B6")
        axes[0].plot(epochs, mean_acc, label=run_name, color=color, linewidth=2.0)
        axes[0].fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)
        
    axes[0].set_title(f"Test Accuracy vs Epochs ({args.task.upper()})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    
    # Test Loss
    for run_name, run_data in results["runs"].items():
        test_losses = np.array(run_data["test_loss_seeds"]) # [num_seeds, num_epochs]
        epochs = range(1, test_losses.shape[1] + 1)
        mean_loss = np.mean(test_losses, axis=0)
        std_loss = np.std(test_losses, axis=0)
        
        color = colors.get(run_name, "#9B59B6")
        axes[1].plot(epochs, mean_loss, label=run_name, color=color, linewidth=2.0)
        axes[1].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)
        
    axes[1].set_title(f"Test Loss vs Epochs ({args.task.upper()})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross Entropy Loss")
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    plot_acc_path = os.path.join(args.artifact_dir, f"learning_curves_{args.task}.png")
    plt.savefig(plot_acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves plot to: {plot_acc_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Total-Compute Efficiency Curve (Accuracy vs Cumulative FLOPs)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for run_name, run_data in results["runs"].items():
        test_accs = np.array(run_data["test_acc_seeds"])
        mean_acc = np.mean(test_accs, axis=0)
        
        mean_init_flops = np.mean(run_data["init_flops_seeds"])
        train_flops_per_epoch = run_data["train_flops_per_epoch"]
        
        # Calculate cumulative FLOPs at each epoch evaluation point
        # Epoch 0 (pre-training initialization FLOPs) -> Epoch E
        num_epochs = len(mean_acc)
        cumulative_flops = [mean_init_flops + e * train_flops_per_epoch for e in range(1, num_epochs + 1)]
        
        color = colors.get(run_name, "#9B59B6")
        plt.plot(cumulative_flops, mean_acc, label=run_name, color=color, linewidth=2.0, marker='o', markersize=4)
        
    plt.title(f"Compute Efficiency: Test Accuracy vs Cumulative FLOPs ({args.task.upper()})")
    plt.xlabel("Total FLOPs (Log Scale)")
    plt.ylabel("Test Accuracy")
    plt.xscale("log")
    plt.legend(loc="lower right")
    
    plot_compute_path = os.path.join(args.artifact_dir, f"compute_efficiency_{args.task}.png")
    plt.savefig(plot_compute_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved compute efficiency plot to: {plot_compute_path}")
    
    # -------------------------------------------------------------
    # Plot 3: Step-0 Layer-by-Layer Activation Variance
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for run_name, run_data in results["runs"].items():
        variances_seeds = np.array(run_data["step0_variances_seeds"]) # [num_seeds, num_layers]
        mean_vars = np.mean(variances_seeds, axis=0)
        layers = range(len(mean_vars))
        
        color = colors.get(run_name, "#9B59B6")
        plt.plot(layers, mean_vars, label=run_name, color=color, linewidth=2.0, marker='s', markersize=5)
        
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Ideal Stability Target (1.0)")
    plt.title(f"Step-0 Activation Variance Layer-by-Layer ({args.task.upper()})")
    plt.xlabel("Layer Index (0=Input, L=Output)")
    plt.ylabel("Variance")
    plt.legend(loc="best")
    
    plot_var_path = os.path.join(args.artifact_dir, f"activation_variances_{args.task}.png")
    plt.savefig(plot_var_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved activation variances plot to: {plot_var_path}")
    
    # -------------------------------------------------------------
    # Plot 4: Step-0 Effective Rank (Singular Value Entropy) comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    run_names = []
    mean_ranks = []
    std_ranks = []
    
    for run_name, run_data in results["runs"].items():
        ranks_seeds = run_data["step0_ranks_seeds"] # list of dicts: [ {name: val}, {name: val}, ... ]
        # We look at the first layer weight matrix (e.g. layers.0.weight)
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
        plt.title(f"Step-0 Weight Matrix Effective Rank Comparison ({args.task.upper()})")
        plt.tight_layout()
        
        plot_rank_path = os.path.join(args.artifact_dir, f"singular_values_{args.task}.png") # Named singular_values to maintain artifact mapping
        plt.savefig(plot_rank_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved step-0 weight matrix rank plot to: {plot_rank_path}")
    else:
        print("No weight parameters found to calculate effective rank. Skipping Plot 4.")

if __name__ == "__main__":
    main()
