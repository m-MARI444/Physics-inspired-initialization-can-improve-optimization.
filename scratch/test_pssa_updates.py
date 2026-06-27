import torch
import torch.nn as nn
import sys
import os

# Adjust paths to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.causal_model import PSSASimulator
from training.thermodynamics import EquilibriumLock

def test_equilibrium_lock_normalization():
    print("Testing EquilibriumLock normalization...")
    
    # 1. Setup mock parameters of different sizes
    param1 = nn.Parameter(torch.randn(32, 64) * 0.1) # numel = 2048
    param2 = nn.Parameter(torch.randn(128, 256) * 0.1) # numel = 32768
    
    routing_params = [("routing.weight1", param1)]
    worldmodel_params = [("worldmodel.weight2", param2)]
    
    lock = EquilibriumLock(
        routing_params=routing_params,
        worldmodel_params=worldmodel_params,
        lock_strength=0.1
    )
    
    # 2. Capture basis
    lock.capture_basis(step=0, verbose=False)
    assert lock.basis_captured, "Basis capture failed"
    
    # 3. Compute loss
    loss = lock.lock_loss()
    print(f"Captured lock loss: {loss.item():.6f}")
    assert loss.ndim == 0, "Lock loss should be a scalar tensor"
    
    # 4. Check gradients propagate
    loss.backward()
    assert param1.grad is not None, "Gradient did not propagate to param1"
    assert param2.grad is not None, "Gradient did not propagate to param2"
    
    print("EquilibriumLock normalization check: PASSED\n")

def test_causal_simulator_per_sequence_gating():
    print("Testing PSSASimulator per-sequence batch gating...")
    
    # Initialize simulator
    model = PSSASimulator(d_in=6, d_action=6, d_model=64, d_out=6, vocab_size=8)
    model.eval()
    
    # Setup inputs
    batch_size = 3
    time_steps = 5
    
    x = torch.randn(batch_size, time_steps, 6)
    
    # Setup actions to trigger different modes for different batch elements:
    # eps_low = 0.5, eps_high = 1.5
    # Sequence 0: Idle mode (< 0.5)
    # Sequence 1: Local mode (>= 0.5 and < 1.5)
    # Sequence 2: Full causal mode (>= 1.5)
    action = torch.zeros(batch_size, time_steps, 6)
    
    # Sequence 0 action norm = 0.1 (Idle)
    action[0, :, 0] = 0.1
    # Sequence 1 action norm = 1.0 (Local)
    action[1, :, 0] = 1.0
    # Sequence 2 action norm = 2.0 (Full causal)
    action[2, :, 0] = 2.0
    
    # We run the forward pass
    preds_phys, preds_lang, states, imagined, kl, stats, sigma = model(
        x, action, return_kl=True, return_stats=True, return_sigma=True
    )
    
    stats_idle, stats_local, stats_full = stats
    print(f"Stats returned: Idle={stats_idle}, Local={stats_local}, Full={stats_full}")
    
    # Verify events count:
    # Idle events should be batch_size=1 * time_steps=5 = 5
    # Local events should be batch_size=1 * time_steps=5 = 5
    # Full events should be batch_size=1 * time_steps=5 = 5
    assert stats_idle == 5, f"Expected 5 idle events, got {stats_idle}"
    assert stats_local == 5, f"Expected 5 local events, got {stats_local}"
    assert stats_full == 5, f"Expected 5 full events, got {stats_full}"
    
    # Verify outputs are shape consistent
    assert preds_phys.shape == (batch_size, time_steps, 6)
    assert preds_lang.shape == (batch_size, time_steps, 8)
    assert states.shape == (batch_size, time_steps, 64)
    assert imagined.shape == (batch_size, time_steps, 64)
    assert kl.ndim == 0, "KL loss should be a scalar tensor"
    assert sigma.ndim == 0, "Sigma std mean should be a scalar tensor"
    
    print("PSSASimulator per-sequence gating check: PASSED\n")

if __name__ == "__main__":
    test_equilibrium_lock_normalization()
    test_causal_simulator_per_sequence_gating()
    print("All unit tests passed successfully!")
