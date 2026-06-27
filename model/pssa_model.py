import torch
import torch.nn as nn
from model.neuron import PersistentNeuronLayer
from model.routing import ContinuousManifoldRouting
from model.topology import WavePropagation

class PSSAV2(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # PSSA Components
        self.neuron_layer = PersistentNeuronLayer(d_model)
        self.routing = ContinuousManifoldRouting(d_model)
        self.topology = WavePropagation(d_model)
        
        self.out_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x_idx, prev_state=None):
        """
        x_idx: [batch, seq]
        prev_state: [batch, seq, d_model] (optional)
        """
        x = self.embed(x_idx)
        batch, seq, d = x.shape
        
        if prev_state is None:
            prev_state = torch.zeros_like(x)
            
        # 1. Continuous Routing Context
        route_weights = self.routing(x)
        routed_x = torch.bmm(route_weights, x)
        
        # 2. State Evolution (Neuron Update)
        new_state, semantic_proj = self.neuron_layer(prev_state, routed_x)
        
        # 3. Multi-scale Wave Propagation
        wave_state = self.topology(new_state)
        
        # Decode
        logits = self.out_proj(wave_state)
        
        return logits, new_state, semantic_proj
