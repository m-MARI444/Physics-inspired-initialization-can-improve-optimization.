import torch
import torch.nn as nn

class PersistentNeuronLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Gated write mechanism
        self.update_gate = nn.Linear(d_model * 2, d_model)
        self.candidate_layer = nn.Linear(d_model * 2, d_model)
        
        # Minimum gate floor to prevent saturation (alpha_min)
        self.alpha_min = 0.05
        
    def forward(self, state_t, x_t):
        """
        state_t: [batch, seq, d_model]
        x_t: [batch, seq, d_model]
        """
        combined = torch.cat([state_t, x_t], dim=-1)
        
        # Calculate write gate with floor
        write_gate = torch.sigmoid(self.update_gate(combined))
        write_gate = torch.clamp(write_gate, min=self.alpha_min)
        
        # Candidate new state
        candidate_state = torch.tanh(self.candidate_layer(combined))
        
        # Evolve state: s_{t+1} = f(s_t, x_t)
        state_t_plus_1 = (1.0 - write_gate) * state_t + write_gate * candidate_state
        
        # Return new state and the projection for semantic energy (P(s))
        return state_t_plus_1, state_t_plus_1
