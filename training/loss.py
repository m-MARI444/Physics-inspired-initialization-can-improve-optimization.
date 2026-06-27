import torch
import torch.nn as nn

class PSSALoss(nn.Module):
    def __init__(self, lambda_drift=0.1):
        super().__init__()
        self.lambda_drift = lambda_drift
        # Ignore padding token (12)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=12)
        
    def forward(self, logits, targets, semantic_projs):
        """
        logits: [batch, seq, vocab_size]
        targets: [batch, seq]
        semantic_projs: [batch, seq, d_model]
        """
        batch, seq, vocab = logits.shape
        
        # 1. Standard task loss
        task_loss = self.ce_loss(logits.reshape(-1, vocab), targets.reshape(-1))
        
        # 2. Semantic Energy Regularizer: L_drift = lambda * |P(s_t) - P(s_0)|^2
        s_0 = semantic_projs[:, 0, :]
        s_t = semantic_projs[:, -1, :]
        drift_loss = torch.mean((s_t - s_0)**2)
            
        total_loss = task_loss + self.lambda_drift * drift_loss
        
        # 3. Lyapunov Monitor tracking: dV/dt (estimated as Energy change)
        dV_dt = drift_loss.item() 
        
        return total_loss, task_loss.item(), drift_loss.item(), dV_dt
