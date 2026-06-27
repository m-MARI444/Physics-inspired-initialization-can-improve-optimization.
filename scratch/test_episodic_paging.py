import torch
import torch.nn as nn
import sys
import os

# Adjust paths to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.memory_manager import EpisodicMemoryManager
from model.pssa_gpt import PSSAGPT

def test_episodic_memory_manager():
    print("Testing EpisodicMemoryManager components...")
    
    batch = 2
    d_model = 64
    active_capacity = 12
    archive_capacity = 200
    manifold_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    manager = EpisodicMemoryManager(
        d_model=d_model,
        num_timelines=3,
        active_capacity=active_capacity,
        archive_capacity=archive_capacity,
        manifold_dim=manifold_dim
    ).to(device)
    
    # 1. Initialize Memory
    archive_bank, archive_keys, active_lru = manager.init_memory(batch, device)
    
    assert archive_bank.shape == (batch, archive_capacity, 3, d_model)
    assert archive_keys.shape == (batch, archive_capacity, manifold_dim)
    assert active_lru.shape == (batch, active_capacity)
    
    # Setup mock active entity banks and coordinate keys
    E_prev = torch.randn(batch, active_capacity, 3, d_model, device=device) * 0.1
    active_keys = torch.randn(batch, active_capacity, manifold_dim, device=device) * 0.1
    
    # Current input feature
    x_t = torch.randn(batch, d_model, device=device)
    
    # 2. Test dynamic page step swapping
    print("Running page_step swap...")
    E_next, active_keys_next, archive_bank, archive_keys, active_lru = manager.page_step(
        x_t, E_prev, active_keys, archive_bank, archive_keys, active_lru, step=1
    )
    
    assert E_next.shape == (batch, active_capacity, 3, d_model)
    assert active_keys_next.shape == (batch, active_capacity, manifold_dim)
    
    # 3. Test prospective prefetching
    print("Running look-ahead prefetch...")
    timeline2_prev = torch.randn(batch, active_capacity, d_model, device=device)
    E_next, active_keys_next, archive_bank, archive_keys, active_lru = manager.prefetch(
        timeline2_prev, E_next, active_keys_next, archive_bank, archive_keys, active_lru, step=1
    )
    
    assert E_next.shape == (batch, active_capacity, 3, d_model)
    print("EpisodicMemoryManager components: PASSED\n")

def test_pssa_gpt_integration():
    print("Testing PSSAGPT integration with Episodic Memory Paging...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize PSSAGPT model
    model = PSSAGPT(
        vocab_size=100,
        d_model=64,
        num_slots=5,
        tau=0.15,
        num_entities=4,  # num_scopes * num_entities = 3 * 4 = 12 active slots
        routing_temp=0.5,
        num_scopes=3,
        num_layers=2
    ).to(device)
    
    # Setup mock input IDs (tokens)
    batch = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch, seq_len), device=device)
    
    # Run forward pass through the paged GPT model
    print("Running PSSAGPT forward pass...")
    outputs = model(input_ids)
    
    logits = outputs[0]
    recon_loss = outputs[9]
    
    print(f"Logits shape: {logits.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.6f}")
    
    assert logits.shape == (batch, seq_len, 100)
    assert recon_loss.ndim == 0
    
    print("PSSAGPT integration check: PASSED\n")

def test_routing_dynamic_pruning():
    print("Testing dynamic edge pruning in ContinuousManifoldRouting...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.routing import ContinuousManifoldRouting
    
    # Initialize routing with a high pruning threshold to trigger zeros
    routing = ContinuousManifoldRouting(d_model=64, manifold_dim=3, pruning_threshold=0.5).to(device)
    
    # Setup mock node coordinates that are far apart
    # x: [batch=1, nodes=4, d_model=64]
    x = torch.zeros(1, 4, 64, device=device)
    # Set feature values to spread manifold coordinates apart
    x[0, 0, 0] = -10.0
    x[0, 1, 0] = 10.0
    x[0, 2, 1] = -10.0
    x[0, 3, 1] = 10.0
    
    # Forward pass
    routing_weights = routing(x)
    
    print(f"Routing weights:\n{routing_weights[0]}")
    
    # Check that weak connections are exactly zero
    # Self connections will remain non-zero (distance = 0, similarity = 1.0 > 0.5)
    # Distant connections will be clamped to 0.0
    num_zeros = (routing_weights == 0.0).sum().item()
    print(f"Number of pruned connections (exact zeros): {num_zeros}")
    assert num_zeros > 0, "Pruning failed, no connections were zeroed"
    
    print("ContinuousManifoldRouting edge pruning check: PASSED\n")

if __name__ == "__main__":
    test_episodic_memory_manager()
    test_pssa_gpt_integration()
    test_routing_dynamic_pruning()
    print("All episodic memory unit tests passed successfully!")
