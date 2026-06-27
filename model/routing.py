import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousManifoldRouting(nn.Module):
    def __init__(self, d_model, manifold_dim=3, pruning_threshold=0.15):
        super().__init__()
        self.manifold_proj = nn.Linear(d_model, manifold_dim)
        self.routing_scale = nn.Parameter(torch.tensor(1.0))
        self.pruning_threshold = pruning_threshold
        
    def forward(self, x):
        """
        x: [batch, nodes, d_model]
        Returns continuous routing weights [batch, nodes, nodes]
        """
        # Project nodes onto low-dimensional semantic manifold
        manifold_coords = self.manifold_proj(x) # [batch, nodes, manifold_dim]
        
        # Compute pairwise distances in the manifold
        dist = torch.cdist(manifold_coords, manifold_coords, p=2.0) # [batch, nodes, nodes]
        
        # Convert distances to continuous routing affinities (close = high affinity)
        # Using Gaussian-like radial basis function
        routing_weights = torch.exp(-self.routing_scale * (dist ** 2))
        
        # --- DYNAMIC EDGE PRUNING ---
        # Clamp all weak affinities below pruning_threshold to exact zero
        mask = (routing_weights >= self.pruning_threshold).float()
        routing_weights = routing_weights * mask
        
        # Self-routing can be zeroed out or kept. Let's keep it normalized.
        routing_weights = F.normalize(routing_weights, p=1, dim=-1)
        
        return routing_weights
