import torch
import torch.nn as nn

class EpisodicMemoryManager(nn.Module):
    """
    Episodic Memory Manager for the PSSA Architecture.
    Handles dynamic page swapping between GPU VRAM (L2 Active Bank) and GPU VRAM (L3 Episodic Archive).
    Optimized to run fully vectorized on the GPU with zero CPU-GPU synchronization stalls.
    """
    def __init__(self, d_model=64, num_timelines=3, active_capacity=12, archive_capacity=200, manifold_dim=3):
        super().__init__()
        self.d_model = d_model
        self.num_timelines = num_timelines
        self.active_capacity = active_capacity
        self.archive_capacity = archive_capacity
        self.manifold_dim = manifold_dim
        
        # Projection layer to compute query keys in manifold space
        self.q_proj = nn.Linear(d_model, manifold_dim)
        
    def init_memory(self, batch, device):
        """
        Initializes the GPU archive banks, keys, and tracking parameters.
        Returns:
          archive_bank: [batch, archive_capacity, num_timelines, d_model] on GPU
          archive_keys: [batch, archive_capacity, manifold_dim] on GPU
          active_lru:   [batch, active_capacity] last access timestamp on GPU
        """
        # Archive stored on GPU to prevent CPU-GPU transfer bottlenecks
        archive_bank = torch.randn(batch, self.archive_capacity, self.num_timelines, self.d_model, device=device) * 0.1
        archive_keys = torch.randn(batch, self.archive_capacity, self.manifold_dim, device=device) * 0.1
        
        # Track the last step at which each active slot was updated or retrieved
        active_lru = torch.zeros(batch, self.active_capacity, dtype=torch.long, device=device)
        
        return archive_bank, archive_keys, active_lru

    def page_step(self, x_t, E_prev, active_keys, archive_bank, archive_keys, active_lru, step):
        """
        Performs page swapping based on semantic relevance to the current input token.
        Vectorized across batch elements with zero CPU synchronization.
        """
        batch = x_t.size(0)
        device = x_t.device
        
        # Ensure active_keys and archive_keys are float32
        active_keys = active_keys.float()
        archive_keys = archive_keys.float()
        
        # 1. Project current token input onto the manifold coordinate space
        q_t = self.q_proj(x_t).float()  # [batch, manifold_dim]
        
        # 2. Compute negative L2 distance (similarity) to all active GPU coordinates
        dist_active = torch.norm(active_keys - q_t.unsqueeze(1), p=2, dim=-1)  # [batch, active_capacity]
        sim_active = -dist_active
        
        # 3. Compute similarity to all archived GPU coordinates
        dist_archive = torch.norm(archive_keys - q_t.unsqueeze(1), p=2, dim=-1)  # [batch, archive_capacity]
        sim_archive = -dist_archive
        
        # 4. Vectorized swap logic
        max_archive_val, max_archive_idx = torch.max(sim_archive, dim=-1)  # [batch], [batch]
        
        age = step - active_lru  # [batch, active_capacity]
        eviction_priority = -sim_active + 0.1 * age.float()  # [batch, active_capacity]
        _, evict_idx = torch.max(eviction_priority, dim=-1)  # [batch], [batch]
        
        # Gather sim_active at evict_idx
        sim_active_evict = sim_active.gather(1, evict_idx.unsqueeze(1)).squeeze(1)  # [batch]
        
        # Determine which batch elements should swap
        swap_mask = max_archive_val > sim_active_evict  # [batch] (bool)
        
        # Get active entities and keys at evict_idx
        num_timelines = E_prev.shape[2]
        gpu_entities = E_prev.gather(1, evict_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model))
        gpu_keys = active_keys.gather(1, evict_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim))
        
        # Get archive entities and keys at max_archive_idx
        cpu_entities = archive_bank.gather(1, max_archive_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model))
        cpu_keys = archive_keys.gather(1, max_archive_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim))
        
        # Swap based on swap_mask
        final_gpu_entities = torch.where(swap_mask.view(batch, 1, 1, 1), cpu_entities, gpu_entities)
        final_cpu_entities = torch.where(swap_mask.view(batch, 1, 1, 1), gpu_entities, cpu_entities)
        final_gpu_keys = torch.where(swap_mask.view(batch, 1, 1), cpu_keys, gpu_keys)
        final_cpu_keys = torch.where(swap_mask.view(batch, 1, 1), gpu_keys, cpu_keys)
        
        # Write back to tensors (out-of-place to protect autograd history)
        E_prev = E_prev.scatter(1, evict_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model), final_gpu_entities)
        archive_bank = archive_bank.scatter(1, max_archive_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model), final_cpu_entities)
        active_keys = active_keys.scatter(1, evict_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim), final_gpu_keys)
        archive_keys = archive_keys.scatter(1, max_archive_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim), final_cpu_keys)
        
        # Mark the most similar active entity slot as recently updated/accessed
        best_active_idx = torch.argmax(sim_active, dim=-1)
        active_lru = active_lru.scatter(1, best_active_idx.unsqueeze(1), step)
        
        return E_prev, active_keys, archive_bank, archive_keys, active_lru

    def prefetch(self, timeline2_prev, E_prev, active_keys, archive_bank, archive_keys, active_lru, step):
        """
        Prefetches entities into GPU VRAM using prospective look-aheads (Timeline 2).
        Vectorized across batch elements with zero CPU synchronization.
        """
        batch = timeline2_prev.size(0)
        device = timeline2_prev.device
        
        # Ensure active_keys and archive_keys are float32
        active_keys = active_keys.float()
        archive_keys = archive_keys.float()
        
        # Project prospective Timeline 2 representations to look-ahead coordinate keys
        q_future = self.q_proj(timeline2_prev).float()  # [batch, active_capacity, manifold_dim]
        
        # Compute pairwise distance between future coordinates and archived coordinates
        dist = torch.cdist(q_future, archive_keys, p=2.0)  # [batch, active_capacity, archive_capacity]
        min_dist, best_archive_indices = torch.min(dist, dim=-1)  # [batch, active_capacity], [batch, active_capacity]
        
        num_timelines = E_prev.shape[2]
        
        # Flatten last two dimensions of dist to find the overall most urgent candidate per batch element
        flat_dist = dist.view(batch, -1)  # [batch, active_capacity * archive_capacity]
        min_val, min_idx = torch.min(flat_dist, dim=-1)  # [batch], [batch]
        
        slot_idx = min_idx // self.archive_capacity  # [batch]
        arch_idx = min_idx % self.archive_capacity  # [batch]
        
        # Condition 1: Activation threshold
        prefetch_mask = min_val < 0.3  # [batch] (bool)
        
        # Condition 2: Avoid duplicate prefetching if the entity is already active
        candidate_key = archive_keys.gather(1, arch_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim))  # [batch, 1, manifold_dim]
        active_diffs = torch.norm(active_keys - candidate_key, dim=-1)  # [batch, active_capacity]
        already_active = (active_diffs < 0.05).any(dim=-1)  # [batch] (bool)
        
        # Combine conditions
        should_prefetch = prefetch_mask & (~already_active)  # [batch] (bool)
        
        # Identify the LRU active slot to evict
        evict_idx = torch.argmin(active_lru, dim=-1)  # [batch]
        
        # Get active entities and keys at evict_idx
        gpu_entities = E_prev.gather(1, evict_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model))
        gpu_keys = active_keys.gather(1, evict_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim))
        
        # Get archive entities and keys at arch_idx
        cpu_entities = archive_bank.gather(1, arch_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model))
        cpu_keys = archive_keys.gather(1, arch_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim))
        
        # Swap based on should_prefetch mask
        final_gpu_entities = torch.where(should_prefetch.view(batch, 1, 1, 1), cpu_entities, gpu_entities)
        final_cpu_entities = torch.where(should_prefetch.view(batch, 1, 1, 1), gpu_entities, cpu_entities)
        final_gpu_keys = torch.where(should_prefetch.view(batch, 1, 1), cpu_keys, gpu_keys)
        final_cpu_keys = torch.where(should_prefetch.view(batch, 1, 1), gpu_keys, cpu_keys)
        
        # Write back to tensors (out-of-place to protect autograd history)
        E_prev = E_prev.scatter(1, evict_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model), final_gpu_entities)
        archive_bank = archive_bank.scatter(1, arch_idx.view(batch, 1, 1, 1).expand(-1, -1, num_timelines, self.d_model), final_cpu_entities)
        active_keys = active_keys.scatter(1, evict_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim), final_gpu_keys)
        archive_keys = archive_keys.scatter(1, arch_idx.view(batch, 1, 1).expand(-1, -1, self.manifold_dim), final_cpu_keys)
        
        # Postpone eviction of the newly prefetched entity
        old_lru_val = active_lru.gather(1, evict_idx.unsqueeze(1)).squeeze(1)
        new_lru_val = torch.where(should_prefetch, torch.tensor(step + 2, device=device), old_lru_val)
        active_lru = active_lru.scatter(1, evict_idx.unsqueeze(1), new_lru_val.unsqueeze(1))
        
        return E_prev, active_keys, archive_bank, archive_keys, active_lru
