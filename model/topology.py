import torch
import torch.nn as nn
import torch.nn.functional as F

class WavePropagation(nn.Module):
    def __init__(self, d_model, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        self.wave_transform = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        x: [batch, seq, d_model]
        Propagates information using hierarchical pooling and unpooling
        """
        batch, seq, d_model = x.shape
        
        # Adjust sequence length if not divisible by pool_size
        padding = (self.pool_size - seq % self.pool_size) % self.pool_size
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding))
            seq += padding
            
        # 1. Compress (pool) locally
        # Reshape to apply pooling across sequence dimension
        x_transpose = x.transpose(1, 2) # [batch, d_model, seq]
        
        # Apply average pooling to compress sequence
        pooled = F.avg_pool1d(x_transpose, kernel_size=self.pool_size, stride=self.pool_size)
        
        # 2. Wave transform (global communication on compressed state)
        pooled = pooled.transpose(1, 2) # [batch, seq//pool, d_model]
        wave_state = F.relu(self.wave_transform(pooled))
        
        # 3. Decompress (unpool) back to local
        wave_state_transpose = wave_state.transpose(1, 2) # [batch, d_model, seq//pool]
        unpooled = F.interpolate(wave_state_transpose, size=seq, mode='nearest')
        
        unpooled = unpooled.transpose(1, 2)
        if padding > 0:
            unpooled = unpooled[:, :-padding, :]
            x = x[:, :-padding, :]
            
        # 4. Residual connection (add global wave back to local state)
        return x + unpooled
