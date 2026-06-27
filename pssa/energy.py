import torch
import torch.nn as nn

def compute_prediction_energy(student_activations, teacher_activations, align_layers=None):
    """
    Computes Prediction Energy (E_pred) measuring the difference between student and teacher.
    If align_layers is None, compares the final logits (last activations).
    Otherwise, compares intermediate layers specified by list of indices.
    """
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
            # Ensure spatial shapes match or flatten if they differ (in this project we align matching shapes)
            if s_act.shape == t_act.shape:
                loss += torch.mean((s_act - t_act) ** 2)
                count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=student_activations[-1].device)


def compute_connection_energy(model, eta=1e-4, beta=1e-3):
    """
    Computes Connection Energy (E_conn) reflecting weight magnitude and structural constraints.
    - L2 regularizer: penalizes large weights.
    - Decorrelation regularizer: penalizes correlation between neuron weight vectors (promotes orthogonality).
    """
    loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            # Flatten to 2D matrix (out_features, in_features)
            w = param.view(param.size(0), -1)
            
            # 1. L2 norm penalty (kinetic weight resistance)
            loss += 0.5 * eta * torch.sum(w ** 2)
            
            # 2. Row decorrelation (orthogonality of neuronal weight vectors)
            if w.size(0) > 1:
                # Normalize rows to unit vectors
                row_norms = torch.norm(w, dim=1, keepdim=True) + 1e-8
                w_norm = w / row_norms
                
                # Compute row correlation matrix (out_features, out_features)
                corr = torch.matmul(w_norm, w_norm.t())
                eye = torch.eye(corr.size(0), device=corr.device)
                
                loss += 0.5 * beta * torch.mean((corr - eye) ** 2)
    return loss


def compute_stability_energy(student_activations, alpha=1.0):
    """
    Computes Stability Energy (E_stab) penalizing variation in activation scale across layers.
    Maintains a stable variance of 1.0 for intermediate activations, preventing exploding/vanishing signals.
    """
    loss = 0.0
    variances = []
    
    # Analyze intermediate layer activations (exclude input and final logits)
    for act in student_activations[1:-1]:
        # Flatten spatial dimensions if needed (batch, features)
        act_flat = act.view(act.size(0), -1)
        # Compute variance across batch per feature, then take mean
        feature_vars = torch.var(act_flat, dim=0)
        variances.append(torch.mean(feature_vars))
        
    if not variances:
        return torch.tensor(0.0, device=student_activations[-1].device)
        
    # 1. Penalize consecutive variance changes (Var(l) - Var(l-1))^2
    for i in range(1, len(variances)):
        loss += (variances[i] - variances[i-1]) ** 2
        
    # 2. Penalize absolute deviation of each layer's variance from 1.0
    for var in variances:
        loss += (var - 1.0) ** 2
        
    return alpha * loss


def compute_information_energy(student_activations, mu=1.0):
    """
    Computes Information Energy (E_info) to maximize information flow.
    Penalizes cross-feature correlations within each layer, forcing activation features to be orthogonal.
    """
    loss = 0.0
    count = 0
    
    # Examine intermediate activations (exclude input and final logits)
    for act in student_activations[1:-1]:
        # Flatten spatial dimensions to (batch_size, num_features)
        act_flat = act.view(act.size(0), -1)
        
        batch_size, num_features = act_flat.shape
        if batch_size <= 1 or num_features <= 1:
            continue
            
        # Center features
        act_centered = act_flat - torch.mean(act_flat, dim=0, keepdim=True)
        
        # Covariance matrix (num_features, num_features)
        cov = torch.matmul(act_centered.t(), act_centered) / (batch_size - 1 + 1e-8)
        
        # Convert to correlation matrix (scale-invariant)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(0))
        
        # Penalize off-diagonals (non-identity values)
        eye = torch.eye(num_features, device=corr.device)
        loss += torch.mean((corr - eye) ** 2)
        count += 1
        
    return mu * (loss / count) if count > 0 else torch.tensor(0.0, device=student_activations[-1].device)


def compute_potential_energy(student_model, student_activations, teacher_activations, config):
    """
    Computes the total potential energy V(W) as a weighted sum of all energy terms.
    """
    v_pred = compute_prediction_energy(
        student_activations, 
        teacher_activations, 
        align_layers=config.get('align_layers', None)
    )
    v_conn = compute_connection_energy(
        student_model, 
        eta=config.get('eta', 1e-4), 
        beta=config.get('beta', 1e-3)
    )
    v_stab = compute_stability_energy(
        student_activations, 
        alpha=config.get('alpha', 1.0)
    )
    v_info = compute_information_energy(
        student_activations, 
        mu=config.get('mu', 1.0)
    )
    
    total_potential = (
        config.get('lambda_pred', 1.0) * v_pred +
        config.get('lambda_conn', 1.0) * v_conn +
        config.get('lambda_stab', 1.0) * v_stab +
        config.get('lambda_info', 1.0) * v_info
    )
    
    energy_breakdown = {
        "potential_total": total_potential.item(),
        "v_pred": v_pred.item(),
        "v_conn": v_conn.item(),
        "v_stab": v_stab.item(),
        "v_info": v_info.item()
    }
    
    return total_potential, energy_breakdown
