import torch
import torch.optim as optim
from tqdm import tqdm
from pssa.models import clone_model
from pssa.energy import compute_potential_energy

def compute_kinetic_energy(model_curr, model_prev, m=1.0, dt=1.0):
    """
    Computes kinetic energy T between two steps of the path.
    T = 0.5 * m * ||W_k - W_{k-1}||^2 / dt^2
    """
    kinetic = 0.0
    for (name_curr, p_curr), (name_prev, p_prev) in zip(model_curr.named_parameters(), model_prev.named_parameters()):
        if p_curr.requires_grad:
            kinetic += torch.sum((p_curr - p_prev) ** 2)
    return 0.5 * m * kinetic / (dt ** 2)


def optimize_least_action(base_model, teacher_model, data_loader, config):
    """
    Optimizes a path of weights to find the stationary action state W_K.
    
    Args:
        base_model (nn.Module): The initial random student model (W_0).
        teacher_model (nn.Module): The pre-trained teacher model.
        data_loader (DataLoader): Loader providing the initialization data.
        config (dict): Configuration dictionary containing:
            - K (int): number of path steps (W_1 ... W_K)
            - m (float): particle mass for kinetic energy
            - dt (float): time step delta
            - lr (float): learning rate for the path optimizer
            - num_steps (int): number of optimization iterations
            - device (str): 'cpu' or 'cuda'
            - potential energy coefficients (lambda_pred, lambda_conn, etc.)
            
    Returns:
        opt_student (nn.Module): The initialized student model (W_K).
        history (dict): Log of energy values over the optimization path.
    """
    device = config.get('device', 'cpu')
    K = config.get('K', 3)
    m = config.get('m', 1.0)
    dt = config.get('dt', 1.0)
    lr = config.get('lr', 1e-3)
    num_steps = config.get('num_steps', 200)
    
    # Move models to device
    base_model = base_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    # 1. Create the path of models W_0, W_1, ..., W_K
    # W_0 is base_model (frozen)
    path = []
    # Add W_0
    w0 = clone_model(base_model)
    for p in w0.parameters():
        p.requires_grad = False
    path.append(w0)
    
    # Add W_1 ... W_K
    params_to_optimize = []
    for k in range(1, K + 1):
        wk = clone_model(base_model)
        for p in wk.parameters():
            p.requires_grad = True
            params_to_optimize.append(p)
        path.append(wk)
        
    # Create optimizer for path parameters
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    
    history = {
        "action": [],
        "kinetic": [],
        "potential": [],
        "v_pred": [],
        "v_conn": [],
        "v_stab": [],
        "v_info": []
    }
    
    # Iterator over the data
    data_iter = iter(data_loader)
    
    print(f"Optimizing Least Action Path of length K={K} (steps={num_steps})...")
    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        
        # Get next batch of data
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, _ = next(data_iter)
            
        x = x.to(device)
        
        # Get teacher activations
        with torch.no_grad():
            t_out, t_acts = teacher_model(x)
            
        total_action = torch.tensor(0.0, device=device)
        total_kinetic = torch.tensor(0.0, device=device)
        total_potential = torch.tensor(0.0, device=device)
        
        step_breakdowns = {
            "v_pred": 0.0,
            "v_conn": 0.0,
            "v_stab": 0.0,
            "v_info": 0.0
        }
        
        # Accumulate kinetic and potential energies along the path
        for k in range(1, K + 1):
            # Kinetic energy T(W_k, W_{k-1})
            t_val = compute_kinetic_energy(path[k], path[k-1], m=m, dt=dt)
            
            # Forward pass through student at step k
            s_out_k, s_acts_k = path[k](x)
            
            # Potential energy V(W_k)
            v_val, breakdown = compute_potential_energy(path[k], s_acts_k, t_acts, config)
            
            # Action component: T + dt * V
            total_action = total_action + t_val + dt * v_val
            total_kinetic = total_kinetic + t_val
            total_potential = total_potential + v_val
            
            for key in step_breakdowns:
                step_breakdowns[key] += breakdown[key]
                
        # Backpropagation
        total_action.backward()
        
        # Gradient clipping to prevent instability in deep path graphs
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
        
        optimizer.step()
        
        # Record history (averaged over path length where appropriate)
        history["action"].append(total_action.item())
        history["kinetic"].append(total_kinetic.item() / K)
        history["potential"].append(total_potential.item() / K)
        for key in step_breakdowns:
            history[key].append(step_breakdowns[key] / K)
            
        pbar.set_postfix({
            "Action": f"{total_action.item():.4f}",
            "T": f"{(total_kinetic.item() / K):.4f}",
            "V": f"{(total_potential.item() / K):.4f}"
        })
        
    # Return the final model W_K and the logs
    opt_student = clone_model(path[-1])
    return opt_student, history
