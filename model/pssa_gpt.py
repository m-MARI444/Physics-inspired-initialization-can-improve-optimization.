import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_manager import EpisodicMemoryManager

class PSSALayer(nn.Module):
    def __init__(self, d_model=64, num_slots=5, tau=0.15, num_entities=4, routing_temp=0.5, num_scopes=3, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.tau = tau
        self.num_entities = num_entities
        self.routing_temp = routing_temp
        self.num_scopes = num_scopes
        self.layer_idx = layer_idx
        
        # --- LAYER-SPECIFIC INERTIA ---
        # Enforces hierarchical temporal cognition: lower = fast, top = stable
        multiplier = 0.5 if layer_idx == 0 else (1.0 if layer_idx == 1 else 2.0)
        self.register_buffer('base_inertia', torch.tensor([0.0, 0.3, 0.7, 1.2, 2.5]) * multiplier)
        self.inertia_scale = 1.0
        self.inertia_momentum = 0.9
        
        self.pred_net = nn.Sequential(
            nn.Linear(num_slots * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.gate_proj = nn.Linear(d_model, num_slots)
        
        self.update_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Project spatial dimensions if coming from below
        self.xt_proj = nn.Linear(num_slots * d_model, d_model) if layer_idx > 0 else nn.Identity()
        
        self.entity_route = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.isolate_gate = nn.Linear(d_model, d_model)
        
        self.scope_net = nn.Sequential(
            nn.Linear(num_slots * d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_scopes)
        )
        
        self.num_timelines = 3
        # Each layer has its own set of memory banks for hierarchical persistence and counterfactual versioning
        self.register_buffer('init_entity_banks', torch.randn(num_scopes, num_entities, self.num_timelines, d_model) * 0.1)
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wave_norm = nn.LayerNorm(d_model)
        
        # STAGE 25: Surprise-Gated Epistemic Inertia bounds
        self.base_inertia = nn.Parameter(torch.ones(num_slots) * 0.10)
        self.inertia_scale = nn.Parameter(torch.ones(1) * 0.25)
        
        # Stage 27 — Distributed Entity Dependency Tracking
        self.wq_dep = nn.Linear(d_model, d_model // 4)
        self.wk_dep = nn.Linear(d_model, d_model // 4)
        
        # Stage 28 — Confidence-Weighted Causal Trust
        self.w_trust = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w_trust.weight)
        nn.init.constant_(self.w_trust.bias, 5.0)
        
        # Stage 29 — Active Counterfactual Imagination & Inference
        self.wq_cf = nn.Linear(d_model, d_model)
        self.wk_cf = nn.Linear(d_model, d_model)
        self.wv_cf = nn.Linear(d_model, d_model)
        self.w_blend = nn.Linear(1, 1)
        nn.init.zeros_(self.wq_cf.weight)
        nn.init.zeros_(self.wq_cf.bias)
        nn.init.zeros_(self.wk_cf.weight)
        nn.init.zeros_(self.wk_cf.bias)
        nn.init.zeros_(self.wv_cf.weight)
        nn.init.zeros_(self.wv_cf.bias)
        nn.init.zeros_(self.w_blend.weight)
        nn.init.constant_(self.w_blend.bias, -5.0)
        
        # Stage 30 — Active Future Simulation & Planning
        self.w_prospective = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.w_prospective.weight)
        nn.init.zeros_(self.w_prospective.bias)
        self.last_cf_attn = None

    def priority_weights(self, S_modules, temperature=1.0, suppress_threshold=0.05):
        probs = F.softmax(S_modules / temperature, dim=-1)
        mask = (probs > suppress_threshold).float()
        masked_weights = probs * mask
        row_sums = masked_weights.sum(dim=-1, keepdim=True) + 1e-8
        w_priority = (masked_weights / row_sums) * 8.0
        return w_priority

    def forward(self, x_t, S_prev, E_prev, V_prev, A_prev, H_prev, token_id=None, actor_mask=None, top_down_precision=None, top_down_surprise=None,
                trust_prev=None, recency_prev=None, mu_S_prev=None, sigma_S_prev=None, cooldown_prev=None,
                fatigue_prev=None, cosine_drift_prev=None, trust_div_prev=None, paradigm_pressure_prev=None,
                decompose_triggered=None):
        batch = S_prev.size(0)
        
        if trust_prev is None:
            trust_prev = torch.ones(batch, 8, device=E_prev.device)
        if recency_prev is None:
            recency_prev = torch.zeros(batch, 8, device=E_prev.device)
        if mu_S_prev is None:
            mu_S_prev = torch.ones(batch, 1, device=E_prev.device) * 0.20
        if sigma_S_prev is None:
            sigma_S_prev = torch.ones(batch, 1, device=E_prev.device) * 0.05
        if cooldown_prev is None:
            cooldown_prev = torch.zeros(batch, 1, device=E_prev.device)
        if fatigue_prev is None:
            fatigue_prev = torch.zeros(batch, 1, device=E_prev.device)
        if cosine_drift_prev is None:
            cosine_drift_prev = torch.zeros(batch, 1, device=E_prev.device)
        if trust_div_prev is None:
            trust_div_prev = torch.zeros(batch, 1, device=E_prev.device)
        if paradigm_pressure_prev is None:
            paradigm_pressure_prev = torch.zeros(batch, 1, device=E_prev.device)
        
        if self.layer_idx == 0:
            x_t_feat = x_t
            x_t_expanded = x_t.unsqueeze(1).expand(-1, self.num_slots, -1)
        else:
            x_t_flat = x_t.view(batch, -1)
            x_t_feat = self.xt_proj(x_t_flat)
            x_t_expanded = x_t
            
        scope_input = torch.cat([S_prev.view(batch, -1), x_t_feat], dim=-1)
        scope_logits = self.scope_net(scope_input)
        scope_logits = torch.clamp(scope_logits, -30.0, 30.0)
        w_t = F.softmax(scope_logits / 0.25, dim=-1)
        
        S_flat = S_prev.view(batch, -1)
        hat_x_t = self.pred_net(S_flat)
        hat_x_t = torch.clamp(hat_x_t, -20.0, 20.0)  # pred_net can amplify large S_prev → Inf surprise
        
        surprise = torch.abs(x_t_feat - hat_x_t)
        mean = surprise.mean(dim=-1, keepdim=True)
        std = torch.sqrt(surprise.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        norm_surprise = (surprise - mean) / std
        norm_surprise = torch.clamp(norm_surprise, -5.0, 5.0)
        
        # Calculate dynamic curiosity-guided surprise index
        if top_down_surprise is not None:
            S_t = torch.tanh(2.0 * top_down_surprise.mean(dim=1)) # [batch, 1]
        else:
            S_t = torch.tanh(2.0 * surprise.mean(dim=-1, keepdim=True)) # [batch, 1]
            
        # Exploratory temperature routing for slot gates
        routing_temp_dyn = self.routing_temp + 0.15 * S_t
        gate_logits = self.gate_proj(norm_surprise)
        gate_logits = torch.clamp(gate_logits, -30.0, 30.0)
        g_t_comp = F.softmax(gate_logits / routing_temp_dyn, dim=-1)
        
        threshold = self.base_inertia.unsqueeze(0) + self.inertia_scale * V_prev
        g_t_surprise = torch.sigmoid(gate_logits - threshold)
        
        g_t = g_t_comp * g_t_surprise
        g_t_expanded = g_t.unsqueeze(-1)
        
        update_input = torch.cat([S_prev, x_t_expanded], dim=-1)
        update_input = torch.nan_to_num(update_input, nan=0.0, posinf=5.0, neginf=-5.0)
        U_t = self.update_net(update_input)
        U_t = torch.nan_to_num(U_t, nan=0.0, posinf=5.0, neginf=-5.0)
        
        S_prime = g_t_expanded * U_t + (1 - g_t_expanded) * S_prev
        S_prime = torch.clamp(S_prime, -10.0, 10.0)
        
        query_input = torch.cat([S_prime[:, 2, :], S_prime[:, 0, :], S_prime[:, 3, :], x_t_feat], dim=-1)
        query_input = torch.nan_to_num(query_input, nan=0.0, posinf=5.0, neginf=-5.0)
        entity_query = self.entity_route(query_input).unsqueeze(1)
        
        # --- PERSISTENT SEMANTIC OBJECT CONSOLIDATION ---
        u_t = U_t[:, 2, :] # [batch, d_model]
        E_prev_active = E_prev[:, :, :, 0, :] # [batch, num_scopes, num_entities, d_model] - Timeline 0: Active World-State
        Timeline2_prev = E_prev[:, :, :, 2, :] # [batch, num_scopes, num_entities, d_model] - Timeline 2: Prospective Buffer
        E_flat = E_prev_active.view(batch, self.num_scopes * self.num_entities, self.d_model)
        
        # Compute semantic similarity between current token update and active object slots
        similarity = torch.bmm(u_t.unsqueeze(1), E_flat.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        
        # Exploratory temperature routing for entity allocation
        alloc_temp_dyn = 0.15 + 0.10 * S_t
        similarity = torch.clamp(similarity, -30.0, 30.0)
        alloc_soft = F.softmax(similarity / alloc_temp_dyn, dim=-1)
        
        # Calculate Shannon Entropy of allocation similarity to capture contextual ambiguity
        h_alloc = - (alloc_soft * torch.log2(alloc_soft + 1e-9)).sum(dim=-1, keepdim=True) # [batch, 1]
        alpha = torch.sigmoid(2.0 * (h_alloc - 1.2)) # [batch, 1]
        
        # Focused Top-1 Allocation (single interpretation prior)
        _, top1_inds = torch.topk(alloc_soft, 1, dim=-1)
        alloc_hard1 = torch.zeros_like(alloc_soft).scatter_(-1, top1_inds, 1.0)
        alloc_scores1 = alloc_hard1 + alloc_soft - alloc_soft.detach()
        
        # Parallel Top-2 Allocation (dual hypothesis tracking)
        _, top2_inds = torch.topk(alloc_soft, 2, dim=-1)
        alloc_hard2 = torch.zeros_like(alloc_soft).scatter_(-1, top2_inds, 1.0)
        alloc_scores2 = alloc_hard2 + alloc_soft - alloc_soft.detach()
        alloc_scores2 = alloc_scores2 / (alloc_scores2.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Continuously interpolate between focused and parallel routing based on ambiguity index
        alloc_scores = alpha * alloc_scores2 + (1.0 - alpha) * alloc_scores1
        
        # Reshape allocation vector
        alloc_scores_expanded = alloc_scores.view(batch, self.num_scopes, self.num_entities, 1)
        
        # Multi-scale decay factors based on layer depth
        if self.layer_idx == 0:
            decay_factor = 0.90
        elif self.layer_idx == 1:
            decay_factor = 0.99
        else:
            decay_factor = 0.9999
            
        # Hierarchical consolidation thresholds modulated by top-down precision and surprise feedback
        S_t_flat = S_t.squeeze(-1) # [batch]
        if self.layer_idx == 0:
            merge_threshold = torch.ones(batch, device=E_prev_active.device)
        elif self.layer_idx == 1:
            base_threshold = 0.95
            if top_down_precision is not None:
                prec_mean = top_down_precision.mean(dim=(1, 2)) # [batch]
                merge_threshold = base_threshold - 0.08 * prec_mean - 0.15 * S_t_flat
            else:
                merge_threshold = base_threshold - 0.15 * S_t_flat
            merge_threshold = torch.clamp(merge_threshold, min=0.0, max=1.0)
        else:
            base_threshold = 0.88
            if top_down_precision is not None:
                prec_mean = top_down_precision.mean(dim=(1, 2)) # [batch]
                merge_threshold = base_threshold - 0.08 * prec_mean - 0.15 * S_t_flat
            else:
                merge_threshold = base_threshold - 0.15 * S_t_flat
            merge_threshold = torch.clamp(merge_threshold, min=0.0, max=1.0)
            
        decayed_E = E_prev_active * decay_factor
        
        # Hypothesis Isolation: modulate u_t per register using its existing state E_prev_active
        isolate_gates = torch.sigmoid(self.isolate_gate(E_prev_active)) # [batch, num_scopes, num_entities, d_model]
        u_t_modulated = u_t.unsqueeze(1).unsqueeze(2) * isolate_gates
        
        # Consolidated attribute integration
        update_val = alloc_scores_expanded * u_t_modulated + (1 - alloc_scores_expanded) * decayed_E
        
        # Apply consolidation only during actor steps, decay otherwise
        if token_id is not None and actor_mask is not None:
            is_actor = actor_mask[token_id] # [batch]
            is_actor_expanded = is_actor.view(batch, 1, 1, 1)
            E_next_active_raw = torch.where(is_actor_expanded, update_val, decayed_E)
        else:
            E_next_active_raw = decayed_E
            is_actor = None
            
        # STAGE 34: Controlled Constitutional Plasticity (Dynamic Thawing)
        # Represents: Physical, Identity, Logical, and Social Invariants
        # Epsilon base is 1e-4 (Stage 33), but scales up to 1e-1 under extreme paradigm pressure (Stage 34)
        epsilon_base = 1e-4
        epsilon_dynamic = epsilon_base + 0.10 * paradigm_pressure_prev.unsqueeze(-1)
        E_next_active = E_next_active_raw.clone()
        E_next_active[:, 0, 0:4, :] = (1.0 - epsilon_dynamic) * E_prev_active[:, 0, 0:4, :] + epsilon_dynamic * E_next_active_raw[:, 0, 0:4, :]
            
        # Episodic Event Summarization in Layer 1, Scope 1 (Event Registers)
        # Scope 1 accumulates running summaries of all active token updates
        if self.layer_idx == 1:
            summary_update = 0.95 * decayed_E[:, 1, :, :] + 0.05 * u_t.unsqueeze(1)
            if is_actor is not None:
                scope1_val = torch.where(is_actor.view(batch, 1, 1), summary_update, decayed_E[:, 1, :, :])
            else:
                scope1_val = summary_update
            # Safely stack scopes to construct the updated directory representation
            E_next_active = torch.stack([E_next_active[:, 0, :, :], scope1_val, E_next_active[:, 2, :, :]], dim=1)
            
        E_next_active = torch.clamp(E_next_active, -10.0, 10.0)
                        
        E_t_flat_active = E_next_active.view(batch, self.num_scopes * self.num_entities, self.d_model)
        
        # Active retrieval (Timeline 0): exactly identical to Stage 27!
        hier_logits_active = torch.bmm(entity_query, E_t_flat_active.transpose(1, 2)).squeeze(1) / 0.15 # [batch, 12]
        hier_logits_active = torch.clamp(hier_logits_active, -30.0, 30.0)
        hier_soft_active = F.softmax(hier_logits_active.unsqueeze(1), dim=-1)
        k_bank = 2
        _, topk_bank_inds = torch.topk(hier_soft_active, k_bank, dim=-1)
        hier_hard_active = torch.zeros_like(hier_soft_active).scatter_(-1, topk_bank_inds, 1.0)
        hier_scores_active = hier_hard_active + hier_soft_active - hier_soft_active.detach()
        hier_scores_active = hier_scores_active / (hier_scores_active.sum(dim=-1, keepdim=True) + 1e-6) # [batch, 1, 12]
        
        # Counterfactual retrieval (Timeline 1)
        E_t_flat_counterfactual = E_prev[:, :, :, 1, :].view(batch, self.num_scopes * self.num_entities, self.d_model)
        hier_logits_counterfactual = torch.bmm(entity_query, E_t_flat_counterfactual.transpose(1, 2)).squeeze(1) / 0.15 # [batch, 12]
        hier_logits_counterfactual = torch.clamp(hier_logits_counterfactual, -30.0, 30.0)
        hier_soft_counterfactual = F.softmax(hier_logits_counterfactual.unsqueeze(1), dim=-1)
        _, topk_bank_inds_cf = torch.topk(hier_soft_counterfactual, k_bank, dim=-1)
        hier_hard_counterfactual = torch.zeros_like(hier_soft_counterfactual).scatter_(-1, topk_bank_inds_cf, 1.0)
        hier_scores_counterfactual = hier_hard_counterfactual + hier_soft_counterfactual - hier_soft_counterfactual.detach()
        hier_scores_counterfactual = hier_scores_counterfactual / (hier_scores_counterfactual.sum(dim=-1, keepdim=True) + 1e-6) # [batch, 1, 12]
        
        # Interpolate using surprise-weighted gate
        beta = 0.20 * S_t.unsqueeze(1) # [batch, 1, 1]
        hier_scores = torch.cat([hier_scores_active * (1.0 - beta), hier_scores_counterfactual * beta], dim=-1) # [batch, 1, 24]
        
        # --- EPISODIC COMPETITIVE ACTIVITY VECTOR DYNAMICS ---
        # The first 12 slots correspond to Timeline 0 (Active World-State)
        retrieval_score = hier_scores[:, 0, :self.num_scopes * self.num_entities] # [batch, 12]
        # Dynamically scale decay factor per slot based on its retrieval score
        decay_factors = 0.99 - 0.05 * (1.0 - retrieval_score) # [batch, 12]
        
        A_flat = A_prev.view(batch, self.num_scopes * self.num_entities)
        A_decayed = A_flat * decay_factors
        
        # --- CUMULATIVE HYSTERESIS SEMANTIC STABILIZATION ---
        # Track persistent cumulative stability index per register to prevent rapid ping-ponging
        H_flat = H_prev.view(batch, self.num_scopes * self.num_entities)
        H_next_flat = 0.90 * H_flat + 0.10 * (1.0 - S_t) # [batch, 12]
        H_next = H_next_flat.view(batch, self.num_scopes, self.num_entities)
        
        # Check surprise revision force against cumulative stability barrier
        revision_force = similarity * S_t # [batch, 12]
        # Hysteresis barrier: gate is open when force overcomes cumulative stability
        hysteresis_gate = torch.sigmoid(15.0 * (revision_force - 0.25 * H_flat)) # [batch, 12]
        
        # Hysteresis-weighted surprise-driven branch revival for dormant slots under contradictory context
        revival_boost = 0.40 * S_t * torch.sigmoid(similarity) * hysteresis_gate # [batch, 12]
        
        # Boost activity continuously via precision-weighted relative activation strength & surprise revival
        activity_boost = alloc_scores + retrieval_score + revival_boost
        A_next_flat = torch.clamp(A_decayed + activity_boost, 0.0, 1.0)
        
        # --- DYNAMIC SEMANTIC ECOLOGY MERGING & GC ---
        device = E_next_active.device
        dot_products = torch.bmm(E_t_flat_active, E_t_flat_active.transpose(1, 2))
        norms = torch.norm(E_t_flat_active, dim=-1, keepdim=True) + 1e-8
        cosine_sim = dot_products / torch.bmm(norms, norms.transpose(1, 2))
        
        # Find similar registers (> merge_threshold similarity)
        similarity_mask = cosine_sim > merge_threshold.view(batch, 1, 1)
        tri_mask = torch.triu(torch.ones(self.num_scopes * self.num_entities, self.num_scopes * self.num_entities, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).expand(batch, -1, -1)
        merge_candidates = similarity_mask & tri_mask
        
        # Precision-Weighted Barycenter Merging
        # Merge duplicate features into primary slots using activity-weighted averaging
        A_E = A_next_flat.unsqueeze(-1) * E_t_flat_active # [batch, 12, d_model]
        merge_proj = merge_candidates.float()
        merged_features_sum = torch.bmm(merge_proj, A_E)
        total_weighted_E = A_E + merged_features_sum
        
        merged_activities_sum = torch.bmm(merge_proj, A_next_flat.unsqueeze(-1))
        total_activities = A_next_flat.unsqueeze(-1) + merged_activities_sum
        
        # Safe division: only average where total activity > 1e-5, otherwise keep E_t_flat_active
        E_flat_merged = torch.where(total_activities > 1e-5, total_weighted_E / (total_activities + 1e-8), E_t_flat_active)
        
        # GC Abscission: Clear duplicate slots & inactive slots (activity < 0.10)
        is_duplicate = merge_candidates.any(dim=1)
        is_inactive = A_next_flat < 0.10
        is_clear = is_duplicate | is_inactive
        
        init_flat = self.init_entity_banks[:, :, 0, :].view(1, self.num_scopes * self.num_entities, self.d_model).expand(batch, -1, -1)
        E_flat_final = torch.where(is_clear.unsqueeze(-1), init_flat, E_flat_merged)
        
        # --- DIFFERENTIABLE ENTITY DEPENDENCY TRACKING (CAUSAL PROPAGATION) ---
        E_prev_flat = E_prev_active.view(batch, self.num_scopes * self.num_entities, self.d_model)
        d_dep = self.d_model // 4
        Q_dep = self.wq_dep(E_prev_flat) # [batch, 12, d_dep]
        K_dep = self.wk_dep(E_prev_flat) # [batch, 12, d_dep]
        
        D_logits = torch.bmm(Q_dep, K_dep.transpose(1, 2)) / (d_dep ** 0.5)
        D = torch.sigmoid(D_logits) # [batch, 12, 12] directed dependency strength
        
        # Stage 28 - Confidence-Weighted propagation attenuation & trust weighting
        source_trust = torch.sigmoid(self.w_trust(E_prev_flat)).squeeze(-1) # [batch, 12]
        target_susceptibility = 1.0 - 0.1 * H_next_flat # [batch, 12]
        
        # Attenuate propagation: trust_matrix[i, j] = D[i, j] * source_trust[j] * target_susceptibility[i]
        trust_matrix = D * source_trust.unsqueeze(1) * target_susceptibility.unsqueeze(2)
        
        # Recursive Sparse Decomposition: Causal Isolation
        if decompose_triggered is not None:
            # Force trust_matrix to identity under decomposition to isolate causal pathways
            I_entities = torch.eye(self.num_scopes * self.num_entities, device=trust_matrix.device).unsqueeze(0).expand(batch, -1, -1)
            trust_matrix = torch.where(decompose_triggered.view(batch, 1, 1), I_entities, trust_matrix)
            
        # Propagate causal updates safely using confidence-weighted trust matrix
        delta_E = E_flat_final - E_prev_flat # [batch, 12, d_model]
        prop_E = 0.15 * torch.bmm(trust_matrix, delta_E) # [batch, 12, d_model]
        
        # Unscaled timeline 0 (active world state) incorporating unscaled causal updates and language features
        E_flat_final_unscaled = E_flat_final + prop_E
        E_active_next_unscaled = E_flat_final_unscaled.view(batch, self.num_scopes, self.num_entities, self.d_model)
        
        # --- STAGE 28 EPISODIC COUNTERFACTUAL TIMELINE SNAPSHOT ---
        is_contradiction = (S_t > 0.40).view(batch, 1, 1, 1)
        Timeline1_next = torch.where(is_contradiction, E_prev_active, 0.98 * E_prev[:, :, :, 1, :])
        
        # --- STAGE 29 ACTIVE COUNTERFACTUAL IMAGINATION ---
        E_active_flat_unscaled = E_active_next_unscaled.view(batch, self.num_scopes * self.num_entities, self.d_model)
        E_cf_flat = Timeline1_next.view(batch, self.num_scopes * self.num_entities, self.d_model)
        
        Q_cf = self.wq_cf(E_active_flat_unscaled)
        K_cf = self.wk_cf(E_cf_flat)
        V_cf = self.wv_cf(E_cf_flat)
        
        A_cf_logits = torch.bmm(Q_cf, K_cf.transpose(1, 2)) / (self.d_model ** 0.5)
        A_cf_logits = torch.clamp(A_cf_logits, -30.0, 30.0)
        A_cf = F.softmax(A_cf_logits, dim=-1)
        self.last_cf_attn = A_cf.detach()
        
        CF_context = torch.bmm(A_cf, V_cf)
        gate_cf = torch.sigmoid(self.w_blend(S_t))
        
        # We compute all intrinsic saliencies and S_modules here (before priority routing blends are applied)
        S_intrinsic_perc = S_t # [batch, 1]
        S_intrinsic_aff = g_t.mean(dim=-1, keepdim=True) # [batch, 1]
        S_intrinsic_lang = (x_t_feat.norm(dim=-1, keepdim=True) / 8.0).clamp(0.0, 1.0) # [batch, 1]
        
        S_intrinsic_mem = A_cf.mean(dim=(-1, -2)).unsqueeze(-1) if A_cf is not None else torch.zeros(batch, 1, device=x_t.device)
        
        cos_sim_plan = F.cosine_similarity(E_active_next_unscaled, E_prev[:, :, :, 2, :], dim=-1).mean(dim=(-1, -2)).unsqueeze(-1)
        S_intrinsic_plan = (1.0 - cos_sim_plan).clamp(0.0, 1.0)
        
        S_intrinsic_affec = (E_active_next_unscaled - E_prev_active).pow(2).mean(dim=(-1, -2, -3)).unsqueeze(-1).clamp(0.0, 1.0)
        S_intrinsic_exec = surprise.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        S_intrinsic_meta = is_contradiction.float().mean(dim=(-1, -2, -3)).unsqueeze(-1) if isinstance(is_contradiction, torch.Tensor) else torch.zeros(batch, 1, device=x_t.device)
        
        S_intrinsic_all = torch.cat([
            S_intrinsic_perc, S_intrinsic_aff, S_intrinsic_lang, S_intrinsic_mem,
            S_intrinsic_plan, S_intrinsic_affec, S_intrinsic_exec, S_intrinsic_meta
        ], dim=-1)
        
        trust_new = trust_prev.clone()
        trust_new[:, 0] = 0.95 * trust_prev[:, 0] + 0.05 * (1.0 - S_intrinsic_perc.squeeze(-1))
        trust_new[:, 1] = 0.95 * trust_prev[:, 1] + 0.05 * (1.0 - S_intrinsic_aff.squeeze(-1))
        trust_new[:, 2] = 0.95 * trust_prev[:, 2] + 0.05 * S_intrinsic_lang.squeeze(-1)
        trust_new[:, 3] = 0.95 * trust_prev[:, 3] + 0.05 * (1.0 - S_intrinsic_mem.squeeze(-1))
        trust_new[:, 4] = 0.95 * trust_prev[:, 4] + 0.05 * cos_sim_plan.squeeze(-1)
        trust_new[:, 5] = 0.95 * trust_prev[:, 5] + 0.05 * (1.0 - S_intrinsic_affec.squeeze(-1))
        trust_new[:, 6] = 0.95 * trust_prev[:, 6] + 0.05 * (1.0 - S_intrinsic_exec.squeeze(-1))
        trust_new[:, 7] = 0.95 * trust_prev[:, 7] + 0.05 * (1.0 - S_intrinsic_meta.squeeze(-1))
        trust_new = trust_new.clamp(0.1, 1.0)
        
        boost = torch.zeros(batch, 8, device=x_t.device)
        is_contra_float = is_contradiction.float().squeeze(-1).squeeze(-1).squeeze(-1)
        boost[:, 0] = is_contra_float * 0.8  # Boost perception
        boost[:, 4] = is_contra_float * 0.8  # Boost planning
        
        S_modules = trust_new * S_intrinsic_all + 0.3 * boost - 0.4 * recency_prev
        S_modules[:, 6] -= fatigue_prev.squeeze(-1) * 0.5
        S_modules[:, 7] -= fatigue_prev.squeeze(-1) * 0.5
        
        # --- PRIORITY ROUTING CONTROLLER ---
        w_priority = self.priority_weights(S_modules) # [batch, 8]
        
        # --- APPLY PRIORITY ROUTING BLENDS TO MODULES ---
        # 1. Perception/Surprise (Module 0): scale S_t
        S_t = S_t * w_priority[:, 0].unsqueeze(-1)
        
        # 2. Affective/Causal Propagation (Module 5): scale prop_E
        prop_E = prop_E * w_priority[:, 5].view(batch, 1, 1)
        E_flat_final = E_flat_final + prop_E
        E_active_next = E_flat_final.view(batch, self.num_scopes, self.num_entities, self.d_model)
        
        # 3. Language Integration (Module 2): scale language update to active state
        E_active_next = E_prev_active + (E_active_next - E_prev_active) * w_priority[:, 2].view(batch, 1, 1, 1)
        
        # 4. Memory (Module 3): scale counterfactual blending
        E_active_flat = E_active_next.view(batch, self.num_scopes * self.num_entities, self.d_model)
        E_active_flat_final = E_active_flat + 0.05 * gate_cf.unsqueeze(-1) * CF_context * w_priority[:, 3].view(batch, 1, 1)
        E_active_next_final = E_active_flat_final.view(batch, self.num_scopes, self.num_entities, self.d_model)
        
        # 5. Planning (Module 4): scale prospective planning
        Timeline2_project = self.w_prospective(E_active_next_final) * w_priority[:, 4].view(batch, 1, 1, 1)
        
        Timeline2_next = torch.where(
            is_contradiction,
            E_prev[:, :, :, 2, :],
            0.95 * E_prev[:, :, :, 2, :] + 0.05 * Timeline2_project
        )
        
        # STAGE 35: Sandbox Timeline (T3)
        if E_prev.size(3) > 3:
            Timeline3_next = E_prev[:, :, :, 3, :].clone()
        else:
            Timeline3_next = Timeline2_next.clone()
        
        # --- HIERARCHICAL REGIONAL WORKSPACE ARBITRATION ---
        # 1. Sensory Workspace (Perception [0] vs Affordance [1])
        sensory_scores = S_modules[:, [0, 1]]
        sensory_winner_idx = torch.argmax(sensory_scores, dim=-1) # [batch]
        sensory_sal = torch.gather(sensory_scores, 1, sensory_winner_idx.unsqueeze(-1)) # [batch, 1]
        
        # 2. Semantic Workspace (Language [2] vs Memory [3])
        semantic_scores = S_modules[:, [2, 3]]
        semantic_winner_idx = torch.argmax(semantic_scores, dim=-1)
        semantic_sal = torch.gather(semantic_scores, 1, semantic_winner_idx.unsqueeze(-1))
        
        # 3. Planning Workspace (Planning [4] vs Affective [5])
        planning_scores = S_modules[:, [4, 5]]
        planning_winner_idx = torch.argmax(planning_scores, dim=-1)
        planning_sal = torch.gather(planning_scores, 1, planning_winner_idx.unsqueeze(-1))
        
        # 4. Executive Workspace (Executive [6] vs Meta [7])
        exec_scores = S_modules[:, [6, 7]]
        exec_winner_idx = torch.argmax(exec_scores, dim=-1)
        exec_sal = torch.gather(exec_scores, 1, exec_winner_idx.unsqueeze(-1))
        
        # --- GLOBAL WORKSPACE COMPETITION (ARBITRATION) ---
        global_sals = torch.cat([sensory_sal, semantic_sal, planning_sal, exec_sal], dim=-1)
        global_winner_idx = torch.argmax(global_sals, dim=-1) # [batch]
        
        # Map regional winner indices back to absolute module indices (vectorized)
        offsets = torch.tensor([0, 2, 4, 6], device=x_t.device)[global_winner_idx]
        local_winners = torch.stack([
            sensory_winner_idx,
            semantic_winner_idx,
            planning_winner_idx,
            exec_winner_idx
        ], dim=1) # [batch, 4]
        winning_module = local_winners[torch.arange(batch, device=x_t.device), global_winner_idx] + offsets
                
        # Enforce Attentional Rotation via Recency Penalties (vectorized)
        winning_mask = F.one_hot(winning_module, num_classes=8).bool() # [batch, 8]
        recency_new = torch.where(winning_mask, recency_prev + 1.0, 0.90 * recency_prev)
        
        # STAGE 33: Introspection Fatigue Dynamics (vectorized)
        is_introspection = (winning_module >= 6).unsqueeze(-1) # [batch, 1]
        is_external = (winning_module <= 1).unsqueeze(-1) # [batch, 1]
        
        fatigue_new = fatigue_prev.clone()
        fatigue_new = torch.where(is_introspection, fatigue_new + 0.1, fatigue_new)
        fatigue_new = torch.where(is_external, torch.clamp(fatigue_new - 0.2, min=0.0), fatigue_new)
                
        # STAGE 33: Multi-dimensional Drift Tracking
        # 1. Cosine Drift: Semantic displacement from invariant Constitution
        constitution_mean = E_next_active[:, 0, 0:4, :].mean(dim=1)
        active_mean = E_next_active.mean(dim=(1, 2))
        curr_cosine_dist = 1.0 - F.cosine_similarity(active_mean, constitution_mean, dim=-1).unsqueeze(-1)
        cosine_drift_new = 0.99 * cosine_drift_prev + 0.01 * curr_cosine_dist
        
        # 2. Trust Divergence: Dispersion of trust weights across modules
        curr_trust_div = trust_new.std(dim=-1, keepdim=True)
        trust_div_new = 0.99 * trust_div_prev + 0.01 * curr_trust_div
                    
        # STAGE 34 & 35: Paradigm Pressure Accumulator & Distributed Consensus
        # If active state contradicts constitution (high drift) AND Perception, Memory, AND Planning trust are high.
        # This prevents an adversarial visual attack from unilaterally rewriting the constitution.
        perc_trust = trust_new[:, 0:1] # [batch, 1]
        mem_trust = trust_new[:, 3:4]
        plan_trust = trust_new[:, 4:5]
        
        # Require quorum for a paradigm shift
        is_paradigm_shift = (curr_cosine_dist > 0.10) & (perc_trust > 0.60) & (mem_trust > 0.60) & (plan_trust > 0.60)
        
        paradigm_pressure_new = 0.95 * paradigm_pressure_prev + 0.05 * is_paradigm_shift.float()
        # Decay pressure slowly if not actively shifting
        paradigm_pressure_new = torch.where(is_paradigm_shift, paradigm_pressure_new, 0.99 * paradigm_pressure_prev)
        
        # --- ADAPTIVE IGNITION THRESHOLDS & ATTENTION COOLDOWN ---
        mu_S_new = 0.90 * mu_S_prev + 0.10 * S_t
        sigma_S_new = torch.sqrt(0.90 * (sigma_S_prev ** 2) + 0.10 * ((S_t - mu_S_new) ** 2) + 1e-5)
        tau_t = mu_S_new + 1.5 * torch.clamp(sigma_S_new, min=0.05)
        
        is_igniting = (S_t > tau_t) & (cooldown_prev == 0)
        
        ign_mask = is_igniting.view(batch, 1)
        cooldown_new = torch.where(ign_mask, torch.ones_like(cooldown_prev) * 2.0, torch.clamp(cooldown_prev - 1.0, min=0.0))
                
        # Global Semantic Synchronization under Conscious Ignition (resets drift)
        sync_mask = is_igniting.view(batch, 1, 1, 1)
        Timeline1_sync = torch.where(sync_mask, Timeline1_next + 0.30 * (E_active_next_final - Timeline1_next), Timeline1_next)
        Timeline2_sync = torch.where(sync_mask, Timeline2_next + 0.30 * (E_active_next_final - Timeline2_next), Timeline2_next)
        
        # STAGE 33: Asynchronous Hybrid Consolidation (Sleep Cycle)
        # Trigger if idle (low surprise) AND chronic drift is pathologically high
        # STAGE 34: Veto sleep if paradigm pressure is high (let the constitution thaw instead!)
        is_sleeping = (S_t < mu_S_new) & (cosine_drift_new > 0.15) & (paradigm_pressure_new < 0.20)
        sleep_mask = is_sleeping.view(batch, 1, 1, 1)
        
        # During sleep: 
        # 1. Prune hallucinated futures back toward stable history
        Timeline2_sync = torch.where(sleep_mask, 0.90 * Timeline2_sync + 0.10 * Timeline1_sync, Timeline2_sync)
        # 2. Reset introspection fatigue
        fatigue_new = torch.where(is_sleeping, torch.zeros_like(fatigue_new), fatigue_new)
        # 3. Pull active state back toward the constitutional invariants
        E_active_next_final = torch.where(sleep_mask, 0.99 * E_active_next_final + 0.01 * constitution_mean.unsqueeze(1).unsqueeze(1), E_active_next_final)
        
        # STAGE 35: Sandbox Hypothesis Routing
        # If there is a localized sensory anomaly (high drift) but NOT full consensus yet,
        # route the simulation into T3 (Sandbox) to see if it predicts the future better.
        anomaly_mask = (curr_cosine_dist > 0.08).view(batch, 1, 1, 1)
        Timeline3_sync = torch.where(anomaly_mask, Timeline3_next + 0.30 * (E_active_next_final - Timeline3_next), Timeline3_next)
        
        # Final Stacked Registry (respecting self.num_timelines)
        if self.num_timelines == 3:
            E_next = torch.stack([E_active_next_final, Timeline1_sync, Timeline2_sync], dim=3)
        else:
            E_next = torch.stack([E_active_next_final, Timeline1_sync, Timeline2_sync, Timeline3_sync], dim=3)
        
        # Reset cleared register activity
        A_next_flat = torch.where(is_clear, torch.ones_like(A_next_flat), A_next_flat)
        A_next = A_next_flat.view(batch, self.num_scopes, self.num_entities)
        
        # Updated flat multi-timeline registry representation (Timeline 0 & 1 only for sequential token matching)
        E_t_flat_all = E_next[:, :, :, :2, :].contiguous().view(batch, self.num_scopes * self.num_entities * 2, self.d_model)
        
        # Scale gates and recalculate S_prime / S_prime_new using Affordance priority (Module 1)
        g_t = g_t * w_priority[:, 1].unsqueeze(-1)
        S_prime = g_t.unsqueeze(-1) * U_t + (1.0 - g_t.unsqueeze(-1)) * S_prev
        S_prime = torch.clamp(S_prime, -10.0, 10.0)
        S_prime_new = S_prime.clone()
        S_prime_new[:, 2, :] = torch.bmm(hier_scores, E_t_flat_all).squeeze(1)
        
        q = self.wq(S_prime_new)
        k = self.wk(S_prime_new)
        v = self.wv(S_prime_new)
        
        A_logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        
        # --- DIFFERENTIABLE TOP-K ROUTING FOR GRAPH ATTENTION ---
        A_soft = torch.sigmoid(A_logits)
        k_slot = 2
        _, topk_slot_inds = torch.topk(A_soft, k_slot, dim=-1)
        A_hard = torch.zeros_like(A_soft).scatter_(-1, topk_slot_inds, 1.0)
        A_sparse = A_hard + A_soft - A_soft.detach()
        row_sums = A_sparse.sum(dim=-1, keepdim=True) + 1e-6
        A_norm = A_sparse / row_sums
        
        # Recursive Sparse Decomposition: Slot Isolation
        if decompose_triggered is not None:
            # Force A_norm to identity to prevent slot-to-slot communication
            I_slots = torch.eye(self.num_slots, device=A_norm.device).unsqueeze(0).expand(batch, -1, -1)
            A_norm = torch.where(decompose_triggered.view(batch, 1, 1), I_slots, A_norm)
        
        A_norm = torch.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.nan_to_num(v, nan=0.0, posinf=5.0, neginf=-5.0)
        wave_update = torch.bmm(A_norm, v)
        wave_update = torch.nan_to_num(wave_update, nan=0.0, posinf=5.0, neginf=-5.0)
        S_res = self.wave_norm(S_prime_new + wave_update)
        S_res = torch.nan_to_num(S_res, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # --- RESIDUAL PERSISTENT ROUTING ---
        if self.layer_idx > 0:
            S_new = x_t + S_res
        else:
            S_new = S_res
            
        S_new = torch.nan_to_num(S_new, nan=0.0, posinf=10.0, neginf=-10.0)
            
        delta_S = torch.norm(S_new - S_prev, dim=-1)
        V_new = self.inertia_momentum * V_prev + (1 - self.inertia_momentum) * delta_S
        
        return (S_new, E_next, V_new, A_next, H_next, g_t, A_norm, hier_scores, U_t, w_t, S_prime_new,
                trust_new, recency_new, mu_S_new, sigma_S_new, cooldown_new, S_modules, winning_module, is_igniting,
                fatigue_new, cosine_drift_new, trust_div_new, paradigm_pressure_new)


class PSSAGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_slots=5, tau=0.15, num_entities=4, routing_temp=0.5, num_scopes=3, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_entities = num_entities
        self.num_scopes = num_scopes
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            PSSALayer(d_model, num_slots, tau, num_entities, routing_temp, num_scopes, layer_idx=l)
            for l in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(num_slots * d_model, vocab_size)
        
        # Top-down Reconstructive decoders between layers (L2 -> L1, L1 -> L0)
        self.recon_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers - 1)
        ])
        
        # Self-assessed slot-wise precision estimators (confidence mapping)
        self.precision_projs = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_layers - 1)
        ])
        
        # Multi-tiered Episodic Memory Manager
        self.memory_manager = EpisodicMemoryManager(
            d_model=d_model,
            num_timelines=3,
            active_capacity=num_scopes * num_entities,
            archive_capacity=200,
            manifold_dim=3
        )

    def forward(self, input_ids, actor_mask=None, k_star_active=2.0, frob_growth_rate=0.0, return_telemetry=True, locked_bases=None, lock_strength=1.0, lock_scale=0.0):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x_emb = self.embed(input_ids)
        # Guard: clamp and fix any NaN or Inf from large embed weights
        x_emb = torch.nan_to_num(x_emb, nan=0.0, posinf=10.0, neginf=-10.0)
        x_emb = torch.clamp(x_emb, -10.0, 10.0)
        
        S_states = [torch.zeros(batch, self.num_slots, self.d_model, device=device) for _ in range(self.num_layers)]
        V_states = [torch.zeros(batch, self.num_slots, device=device) for _ in range(self.num_layers)]
        E_states = [layer.init_entity_banks.unsqueeze(0).expand(batch, -1, -1, -1, -1).clone() for layer in self.layers]
        A_states = [torch.ones(batch, self.num_scopes, self.num_entities, device=device) for _ in range(self.num_layers)]
        H_states = [torch.ones(batch, self.num_scopes, self.num_entities, device=device) for _ in range(self.num_layers)]
        
        # Initialize L3 CPU archives and LRU tracking for each layer
        archive_banks = [None for _ in range(self.num_layers)]
        archive_keys = [None for _ in range(self.num_layers)]
        active_lrus = [None for _ in range(self.num_layers)]
        for l in range(self.num_layers):
            ab, ak, alru = self.memory_manager.init_memory(batch, device)
            archive_banks[l] = ab
            archive_keys[l] = ak
            active_lrus[l] = alru
            
        active_keys = [
            self.memory_manager.q_proj(E_states[l][:, :, :, 0, :].view(batch, self.num_scopes * self.num_entities, self.d_model)).detach().float()
            for l in range(self.num_layers)
        ]
        
        # Cognitive pressure tracking variables
        prev_is_igniting = None
        prev_scope_weights = None
        
        # Stage 31 Cognitive Society Tracking States
        # 8 modules: 0=Perc, 1=Aff, 2=Lang, 3=Mem, 4=Plan, 5=Affec, 6=Exec, 7=Meta
        trust_states = [torch.ones(batch, 8, device=device) for _ in range(self.num_layers)]
        recency_states = [torch.zeros(batch, 8, device=device) for _ in range(self.num_layers)]
        mu_S_states = [torch.ones(batch, 1, device=device) * 0.20 for _ in range(self.num_layers)]
        sigma_S_states = [torch.ones(batch, 1, device=device) * 0.05 for _ in range(self.num_layers)]
        cooldown_states = [torch.zeros(batch, 1, device=device) for _ in range(self.num_layers)]
        
        # STAGE 33 & 34 Epistemic Health States
        fatigue_states = [torch.zeros(batch, 1, device=device) for _ in range(self.num_layers)]
        cosine_drift_states = [torch.zeros(batch, 1, device=device) for _ in range(self.num_layers)]
        trust_div_states = [torch.zeros(batch, 1, device=device) for _ in range(self.num_layers)]
        paradigm_pressure_states = [torch.zeros(batch, 1, device=device) for _ in range(self.num_layers)]
        
        # Initialize top-down precision and surprise feedbacks for step 0
        sigma_0_prev = torch.ones(batch, self.num_slots, 1, device=device) * 0.5
        sigma_1_prev = torch.ones(batch, self.num_slots, 1, device=device) * 0.5
        surprise_0_prev = torch.zeros(batch, self.num_slots, 1, device=device)
        surprise_1_prev = torch.zeros(batch, self.num_slots, 1, device=device)
        
        logits = []
        recon_losses = []
        layer_gates = [[] for _ in range(self.num_layers)]
        layer_slots = [[] for _ in range(self.num_layers)]
        layer_adj = [[] for _ in range(self.num_layers)]
        layer_entities = [[] for _ in range(self.num_layers)]
        layer_entities_full = [[] for _ in range(self.num_layers)]
        layer_retrievals = [[] for _ in range(self.num_layers)]
        layer_writes = [[] for _ in range(self.num_layers)]
        layer_scopes = [[] for _ in range(self.num_layers)]
        layer_pre_wave = [[] for _ in range(self.num_layers)]
        
        # Expose workspace history
        layer_trust = [[] for _ in range(self.num_layers)]
        layer_recency = [[] for _ in range(self.num_layers)]
        layer_saliences = [[] for _ in range(self.num_layers)]
        layer_winners = [[] for _ in range(self.num_layers)]
        layer_ignitions = [[] for _ in range(self.num_layers)]
        
        # Expose epistemic health
        layer_fatigue = [[] for _ in range(self.num_layers)]
        layer_drift = [[] for _ in range(self.num_layers)]
        layer_trust_div = [[] for _ in range(self.num_layers)]
        layer_paradigm = [[] for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            # Calculate cognitive pressure P and decompose_triggered dynamically
            if t > 0 and prev_is_igniting is not None and prev_scope_weights is not None:
                ign_tensor = torch.stack(prev_is_igniting, dim=0) # [num_layers, batch]
                ignition_rate = ign_tensor.float().mean(dim=0) # [batch]
                
                scope_tensor = torch.stack(prev_scope_weights, dim=0) # [num_layers, batch, num_scopes]
                ent = - (scope_tensor * torch.log(scope_tensor + 1e-9)).sum(dim=-1) # [num_layers, batch]
                scope_entropy = ent.mean(dim=0) # [batch]
            else:
                ignition_rate = torch.zeros(batch, device=device)
                scope_entropy = torch.ones(batch, device=device) * 1.0
                
            # Normalize signals element-wise
            p_ignition = torch.clamp(ignition_rate / 0.33, max=1.0)
            p_entropy  = 1.0 - torch.clamp(scope_entropy / 2.0, max=1.0)
            if not isinstance(k_star_active, torch.Tensor):
                k_star_active = torch.tensor(k_star_active, device=device)
            if not isinstance(frob_growth_rate, torch.Tensor):
                frob_growth_rate = torch.tensor(frob_growth_rate, device=device)
                
            p_kstar    = torch.clamp((k_star_active - 2.0).clamp(min=0.0) / 2.0, max=1.0)
            p_frob     = torch.clamp(frob_growth_rate / 0.10, max=1.0)
            
            P_pressure = 0.35 * p_ignition + 0.30 * p_entropy + 0.20 * p_kstar + 0.15 * p_frob
            decompose_triggered = P_pressure > 0.35 # [batch] boolean tensor

            # Initialize step-specific collections for tracking next step's pressure
            current_is_igniting = []
            current_scope_weights = []

            # Detach recurrent states during training to prevent backpropagation through time (BPTT)
            # gradient explosion/NaNs across long sequences in the complex cognitive tracking equations.
            if self.training:
                S_states = [s.detach() for s in S_states]
                E_states = [e.detach() for e in E_states]
                V_states = [v.detach() for v in V_states]
                A_states = [a.detach() for a in A_states]
                H_states = [h.detach() for h in H_states]
                trust_states = [tr.detach() for tr in trust_states]
                recency_states = [re.detach() for re in recency_states]
                mu_S_states = [mu.detach() for mu in mu_S_states]
                sigma_S_states = [sig.detach() for sig in sigma_S_states]
                cooldown_states = [co.detach() for co in cooldown_states]
                fatigue_states = [fa.detach() for fa in fatigue_states]
                cosine_drift_states = [cd.detach() for cd in cosine_drift_states]
                trust_div_states = [td.detach() for td in trust_div_states]
                paradigm_pressure_states = [pp.detach() for pp in paradigm_pressure_states]

            x_t = x_emb[:, t, :]
            token_id = input_ids[:, t]
            
            current_input = x_t
            
            # Bottom-Up Forward pass modulated by top-down precision and surprise feedback priors
            for l, layer in enumerate(self.layers):
                td_prec = None
                td_surp = None
                if l == 0:
                    td_prec = sigma_0_prev
                    td_surp = surprise_0_prev
                elif l == 1:
                    td_prec = sigma_1_prev
                    td_surp = surprise_1_prev
                    
                import torch.utils.checkpoint as checkpoint                
                # Checkpoint the layer forward pass if training to bound VRAM usage.
                # Phase 3 disables checkpointing (model.use_checkpoint=False) because
                # checkpoint.checkpoint recomputes under autograd, which can produce
                # Inf in fp32 for specific inputs even when the forward pass is finite.
                use_ckpt = (self.training and current_input.requires_grad
                            and getattr(self, 'use_checkpoint', True))
                num_timelines = E_states[l].shape[3]
                E_prev_flat = E_states[l].view(batch, self.num_scopes * self.num_entities, num_timelines, self.d_model).clone()
                q_features = current_input if l == 0 else current_input.mean(dim=1)
                
                E_prev_flat_updated, active_keys[l], archive_banks[l], archive_keys[l], active_lrus[l] = self.memory_manager.page_step(
                    q_features, E_prev_flat, active_keys[l], archive_banks[l], archive_keys[l], active_lrus[l], t
                )
                E_states[l] = E_prev_flat_updated.view(batch, self.num_scopes, self.num_entities, num_timelines, self.d_model)
                
                if use_ckpt:
                    (S_new, E_next, V_new, A_new, H_new, g_t, A_norm, hier_scores, U_t, w_t, S_prime_new,
                     trust_new, recency_new, mu_S_new, sigma_S_new, cooldown_new, S_modules, winning_module, is_igniting,
                     fatigue_new, cosine_drift_new, trust_div_new, paradigm_pressure_new) = checkpoint.checkpoint(
                        layer,
                        current_input, S_states[l], E_states[l], V_states[l], A_states[l], H_states[l], token_id, actor_mask,
                        td_prec, td_surp,
                        trust_states[l], recency_states[l],
                        mu_S_states[l], sigma_S_states[l], cooldown_states[l],
                        fatigue_states[l], cosine_drift_states[l], trust_div_states[l], paradigm_pressure_states[l],
                        decompose_triggered,  # Pass decompose_triggered to layer
                        use_reentrant=False
                    )
                else:
                    (S_new, E_next, V_new, A_new, H_new, g_t, A_norm, hier_scores, U_t, w_t, S_prime_new,
                     trust_new, recency_new, mu_S_new, sigma_S_new, cooldown_new, S_modules, winning_module, is_igniting,
                     fatigue_new, cosine_drift_new, trust_div_new, paradigm_pressure_new) = layer(
                        current_input, S_states[l], E_states[l], V_states[l], A_states[l], H_states[l], token_id, actor_mask,
                        top_down_precision=td_prec, top_down_surprise=td_surp,
                        trust_prev=trust_states[l], recency_prev=recency_states[l],
                        mu_S_prev=mu_S_states[l], sigma_S_prev=sigma_S_states[l], cooldown_prev=cooldown_states[l],
                        fatigue_prev=fatigue_states[l], cosine_drift_prev=cosine_drift_states[l], trust_div_prev=trust_div_states[l],
                        paradigm_pressure_prev=paradigm_pressure_states[l],
                        decompose_triggered=decompose_triggered  # Pass decompose_triggered to layer
                    )

                timeline2_flat = E_next[:, :, :, 2, :].view(batch, self.num_scopes * self.num_entities, self.d_model)
                num_timelines = E_next.shape[3]
                E_next_flat = E_next.view(batch, self.num_scopes * self.num_entities, num_timelines, self.d_model).clone()
                
                E_next_flat_prefetched, active_keys[l], archive_banks[l], archive_keys[l], active_lrus[l] = self.memory_manager.prefetch(
                    timeline2_flat, E_next_flat, active_keys[l], archive_banks[l], archive_keys[l], active_lrus[l], t
                )
                E_next = E_next_flat_prefetched.view(batch, self.num_scopes, self.num_entities, num_timelines, self.d_model)

                # Store layer ignition and scope weights for next step pressure calculation
                current_is_igniting.append(is_igniting.squeeze(-1) if is_igniting.dim() > 1 else is_igniting)
                current_scope_weights.append(w_t)

                # ── Inter-layer NaN/Inf firewall ─────────────────────────────
                # S_new flows into the next layer as current_input AND is stored
                # as recurrent state. Any non-finite value here cascades to all
                # subsequent layers and time steps. Clamp it here unconditionally.
                if not torch.isfinite(S_new).all():
                    S_new = torch.nan_to_num(S_new, nan=0.0, posinf=10.0, neginf=-10.0)

                S_states[l] = S_new
                E_states[l] = E_next
                V_states[l] = V_new
                A_states[l] = A_new
                H_states[l] = H_new
                trust_states[l] = trust_new
                recency_states[l] = recency_new
                mu_S_states[l] = mu_S_new
                sigma_S_states[l] = sigma_S_new
                cooldown_states[l] = cooldown_new
                fatigue_states[l] = fatigue_new
                cosine_drift_states[l] = cosine_drift_new
                trust_div_states[l] = trust_div_new
                paradigm_pressure_states[l] = paradigm_pressure_new
                
                if return_telemetry:
                    layer_gates[l].append(g_t.detach())
                    layer_slots[l].append(S_new.detach())
                    layer_adj[l].append(A_norm.detach())
                    layer_entities[l].append(E_next[:, :, :, 0, :].detach())
                    layer_entities_full[l].append(E_next.detach())
                    layer_retrievals[l].append(hier_scores.squeeze(1).detach())
                    layer_writes[l].append(U_t.detach())
                    layer_scopes[l].append(w_t.detach())
                    layer_pre_wave[l].append(S_prime_new.detach())
                    
                    layer_trust[l].append(trust_new.detach())
                    layer_recency[l].append(recency_new.detach())
                    layer_saliences[l].append(S_modules.detach())
                    layer_winners[l].append(winning_module.detach())
                    layer_ignitions[l].append(is_igniting.detach())
                    layer_fatigue[l].append(fatigue_new.detach())
                    layer_drift[l].append(cosine_drift_new.detach())
                    layer_trust_div[l].append(trust_div_new.detach())
                    layer_paradigm[l].append(paradigm_pressure_new.detach())
                
                current_input = S_new
                
            # Cache step outputs for cognitive pressure estimation at step t+1 (detached to prevent memory leaks)
            prev_is_igniting = [i.detach() for i in current_is_igniting]
            prev_scope_weights = [w.detach() for w in current_scope_weights]
                
            # Top-Down Reconstructive Feedback and Precision-Weighted Predictive Coding loop
            # Cast states to float32 to ensure numerical stability in loss and backward pass
            s_state_2_f32 = S_states[2].float()
            s_state_1_f32 = S_states[1].float()
            s_state_0_f32 = S_states[0].float()
            
            hat_S_1 = self.recon_projs[1](s_state_2_f32).float() # L2 -> L1
            hat_S_0 = self.recon_projs[0](s_state_1_f32).float() # L1 -> L0
            
            # Estimate dynamic precision (confidence) per slot in float32
            sigma_1 = torch.sigmoid(self.precision_projs[1](s_state_2_f32)).float() # [batch, num_slots, 1]
            sigma_0 = torch.sigmoid(self.precision_projs[0](s_state_1_f32)).float() # [batch, num_slots, 1]
            
            # Compute slot-wise predictive coding reconstruction surprise (prediction error)
            surprise_1 = (s_state_1_f32 - hat_S_1).pow(2).mean(dim=-1, keepdim=True) # [batch, num_slots, 1]
            surprise_0 = (s_state_0_f32 - hat_S_0).pow(2).mean(dim=-1, keepdim=True) # [batch, num_slots, 1]
            
            # Update precision and surprise feedbacks for the next step (cast back to half if needed, or keep in float since they are detached)
            sigma_0_prev = sigma_0.detach()
            sigma_1_prev = sigma_1.detach()
            surprise_0_prev = surprise_0.detach()
            surprise_1_prev = surprise_1.detach()
            
            # Precision-weighted reconstruction loss with logarithmic regularization to prevent collapse
            # Use larger epsilon 1e-4 for log gradient stability
            loss_1 = (sigma_1 * (s_state_1_f32 - hat_S_1).pow(2)).mean() - 0.01 * torch.log(sigma_1 + 1e-4).mean()
            loss_0 = (sigma_0 * (s_state_0_f32 - hat_S_0).pow(2)).mean() - 0.01 * torch.log(sigma_0 + 1e-4).mean()
            recon_losses.append(loss_1 + loss_0)
            
            # Dynamic Gated Guidance Residual Injection
            # Cast back to original precision type (mixed precision autocast) for residual update.
            # Detach the feedback signal to prevent unstable recurrent backpropagation loops.
            S_states[1] = S_states[1] + (0.05 * sigma_1.detach() * hat_S_1.detach()).to(S_states[1].dtype)
            S_states[0] = S_states[0] + (0.05 * sigma_0.detach() * hat_S_0.detach()).to(S_states[0].dtype)
            
            next_logits = self.out_proj(S_states[-1].view(batch, -1))
            logits.append(next_logits)
            
        logits_tensor = torch.stack(logits, dim=1)
        recon_loss = torch.stack(recon_losses).mean()
        
        if locked_bases is not None and len(locked_bases) > 0:
            scale_val = lock_scale.item() if isinstance(lock_scale, torch.Tensor) else float(lock_scale)
            if scale_val > 0.0:
                total_lock = torch.tensor(0.0, device=device)
                named_params = {name.replace("module.", ""): p for name, p in self.named_parameters()}
                pairs_count = 0
                for name, U in locked_bases.items():
                    if name in named_params:
                        p = named_params[name]
                        if p.dim() >= 2:
                            U_dev = U.to(p.device, dtype=p.dtype)
                            coeff = U_dev.T @ p
                            p_var = p.pow(2).mean()
                            c_var = coeff.pow(2).sum() / p.numel()
                            total_lock = total_lock + (p_var - c_var)
                            pairs_count += 1
                if pairs_count > 0:
                    loss_lock = (lock_strength * scale_val) * (total_lock / pairs_count)
                    recon_loss = recon_loss + loss_lock
        
        if not return_telemetry:
            dummy = torch.zeros(1, device=input_ids.device)
            return (logits_tensor, 
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy,
                    recon_loss,
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy)

        # Save historical multi-timeline states onto model instance for advanced dashboard queries (detached to prevent VRAM accumulation)
        self.last_timeline_states = torch.stack([torch.stack(le, dim=1) for le in layer_entities_full], dim=2).detach()
        
        # Save Stage 31 Global Workspace Society histories onto model instance
        self.last_trust_states = torch.stack([torch.stack(lt, dim=1) for lt in layer_trust], dim=2).detach()
        self.last_recency_states = torch.stack([torch.stack(lr, dim=1) for lr in layer_recency], dim=2).detach()
        self.last_module_saliences = torch.stack([torch.stack(ls, dim=1) for ls in layer_saliences], dim=2).detach()
        self.last_workspace_winners = torch.stack([torch.stack(lw, dim=1) for lw in layer_winners], dim=2).detach()
        self.last_ignitions = torch.stack([torch.stack(li, dim=1) for li in layer_ignitions], dim=2).detach()
        
        out_gates = torch.stack([torch.stack(lg, dim=1) for lg in layer_gates], dim=2)
        out_slots = torch.stack([torch.stack(ls, dim=1) for ls in layer_slots], dim=2)
        out_adj = torch.stack([torch.stack(la, dim=1) for la in layer_adj], dim=2)
        out_entities = torch.stack([torch.stack(le, dim=1) for le in layer_entities], dim=2)
        out_retrievals = torch.stack([torch.stack(lr, dim=1) for lr in layer_retrievals], dim=2)
        out_writes = torch.stack([torch.stack(lw, dim=1) for lw in layer_writes], dim=2)
        out_scopes = torch.stack([torch.stack(lsc, dim=1) for lsc in layer_scopes], dim=2)
        out_pre_wave = torch.stack([torch.stack(lp, dim=1) for lp in layer_pre_wave], dim=2)
        
        out_trust = torch.stack([torch.stack(lt, dim=1) for lt in layer_trust], dim=2)
        out_recency = torch.stack([torch.stack(lr, dim=1) for lr in layer_recency], dim=2)
        out_ignition = torch.stack([torch.stack(li, dim=1) for li in layer_ignitions], dim=2)
        out_fatigue = torch.stack([torch.stack(lf, dim=1) for lf in layer_fatigue], dim=2)
        out_drift = torch.stack([torch.stack(ld, dim=1) for ld in layer_drift], dim=2)
        out_trust_div = torch.stack([torch.stack(ltd, dim=1) for ltd in layer_trust_div], dim=2)
        out_paradigm = torch.stack([torch.stack(lp, dim=1) for lp in layer_paradigm], dim=2)
        
        return (logits_tensor, 
                out_gates.detach(), 
                out_slots.detach(), 
                out_pre_wave.detach(), 
                out_adj.detach(), 
                out_entities.detach(), 
                out_retrievals.detach(), 
                out_writes.detach(), 
                out_scopes.detach(), 
                recon_loss,
                out_trust.detach(), 
                out_recency.detach(), 
                out_ignition.detach(), 
                out_fatigue.detach(), 
                out_drift.detach(), 
                out_trust_div.detach(), 
                out_paradigm.detach())

    def load_state_dict(self, state_dict, strict=True):
        adapted_dict = {}
        for k, v in state_dict.items():
            if "init_entity_banks" in k:
                if v.dim() == 3:
                    # Tiled along timelines to get [num_scopes, num_entities, 3, d_model]
                    v = v.unsqueeze(2).repeat(1, 1, 3, 1)
                elif v.dim() == 4 and v.size(2) == 2:
                    # Tile timeline 2 from timeline 0 to get [num_scopes, num_entities, 3, d_model]
                    t2_init = v[:, :, 0:1, :]
                    v = torch.cat([v, t2_init], dim=2)
            adapted_dict[k] = v
        return super().load_state_dict(adapted_dict, strict=False)
