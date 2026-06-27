import torch
import torch.nn as nn
import torch.nn.functional as F

class PSSALanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_slots=5):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 1. Prediction Network (Predicts next token embedding from current slots)
        self.pred_net = nn.Sequential(
            nn.Linear(num_slots * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 2. Prediction-Error Gate
        self.gate_proj = nn.Linear(d_model, num_slots)
        
        # Base semantic inertia for the 5 hierarchical slots: 
        # Syntax, Entity, Dialogue, Semantic, Attention Focus
        self.register_buffer('base_inertia', torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]))
        self.inertia_momentum = 0.9
        self.inertia_scale = 1.0
        
        # 3. Slot Update Network
        self.update_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 4. Dynamic Relational Wave Propagation
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wave_norm = nn.LayerNorm(d_model)
        
        # 5. Output Projection
        self.out_proj = nn.Linear(num_slots * d_model, vocab_size)
        
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x_emb = self.embed(input_ids) # [batch, seq_len, d_model]
        
        # Initialize Persistent Slots and Velocity (Inertia)
        S_t = torch.zeros(batch, self.num_slots, self.d_model, device=device)
        V_t = torch.zeros(batch, self.num_slots, device=device)
        
        gates = []
        logits = []
        slots = []
        
        for t in range(seq_len):
            x_t = x_emb[:, t, :] # [batch, d_model]
            
            # 1. Prediction (What token does the model expect?)
            S_flat = S_t.view(batch, -1)
            hat_x_t = self.pred_net(S_flat)
            
            # 2. Prediction-Error Gate (How surprised is the model?)
            surprise = torch.abs(x_t - hat_x_t)
            gate_logits = self.gate_proj(surprise)
            
            # Modulate with inertia: g_t = sigmoid(surprise - (base_inertia + velocity))
            threshold = self.base_inertia.unsqueeze(0) + self.inertia_scale * V_t
            g_t = torch.sigmoid(gate_logits - threshold) # [batch, num_slots]
            g_t_expanded = g_t.unsqueeze(-1)
            
            # 3. Slot Update
            x_t_expanded = x_t.unsqueeze(1).expand(-1, self.num_slots, -1)
            update_input = torch.cat([S_t, x_t_expanded], dim=-1)
            U_t = self.update_net(update_input)
            
            S_prime = g_t_expanded * U_t + (1 - g_t_expanded) * S_t
            
            # 4. Dynamic Relational Wave Propagation (Cross-Attention between slots)
            q = self.wq(S_prime)
            k = self.wk(S_prime)
            v = self.wv(S_prime)
            
            attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5), dim=-1)
            wave_update = torch.bmm(attn_weights, v)
            
            S_new = self.wave_norm(S_prime + wave_update)
            
            # 5. Update Velocity (Inertia)
            delta_S = torch.norm(S_new - S_t, dim=-1) # [batch, num_slots]
            V_t = self.inertia_momentum * V_t + (1 - self.inertia_momentum) * delta_S
            
            S_t = S_new
            
            # 6. Predict Next Token Logits
            next_logits = self.out_proj(S_t.view(batch, -1))
            
            gates.append(g_t)
            logits.append(next_logits)
            slots.append(S_t)
            
        gates_tensor = torch.stack(gates, dim=1) # [batch, seq, num_slots]
        logits_tensor = torch.stack(logits, dim=1) # [batch, seq, vocab]
        slots_tensor = torch.stack(slots, dim=1) # [batch, seq, num_slots, d_model]
        
        return logits_tensor, gates_tensor, slots_tensor
