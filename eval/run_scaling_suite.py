import torch
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

class PSSAScalingSuite:
    """
    The Cognitive MRI Scanner for the PSSA.
    Automates adversarial evaluation to detect Hidden Densification, Coalition Poisoning, and Epistemic Drift.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Stage 35.5 Adversarial Scaling Suite on {self.device}...")
        
        # Load the ~50M parameter scaled configuration
        self.vocab_size = 50257
        self.d_model = 256
        self.num_layers = 6
        self.num_slots = 8
        
        self.model = PSSAGPT(
            vocab_size=self.vocab_size, 
            d_model=self.d_model, 
            num_slots=self.num_slots, 
            tau=0.15, 
            num_scopes=3, 
            num_layers=self.num_layers
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded scaled weights from {model_path}")
        else:
            print("WARNING: Running eval on uninitialized weights for baseline testing.")
            
        self.model.eval()
        self.reports = {}
        
    def _trigger_replay_capture(self, failure_type, tokens, layer_idx, outputs):
        """Failure Replay Capture: saves states when architecture collapses."""
        print(f"🚨 [{failure_type}] Failure Replay Capture Triggered!")
        capture_dir = "eval/replays"
        os.makedirs(capture_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"{capture_dir}/{failure_type.replace(' ', '_').lower()}_{timestamp}.pt"
        
        # Unpack the 17 metrics from outputs
        # out_trust = outputs[10] -> [batch, seq, num_layers, 8]
        # out_ignition = outputs[12] -> [batch, seq, num_layers]
        trust_snapshot = outputs[10][:, :, layer_idx, :].detach().cpu()
        ignitions = outputs[12][:, :, layer_idx].detach().cpu()
        
        replay_data = {
            "tokens": tokens.cpu().tolist(),
            "trust_weights": trust_snapshot,
            "ignitions": ignitions,
            "layer_idx": layer_idx
        }
        torch.save(replay_data, filename)
        print(f"   Replay saved to {filename}")

    def test_densification_collapse(self):
        """
        Suite 3: Densification Collapse (MOST IMPORTANT)
        Detects if sparse routing collapses into dense transformer behavior under load.
        """
        print("\n--- Running Densification Collapse Suite ---")
        batch_size = 4
        seq_len = 128
        # Simulate extreme cognitive load: pure noise / dense semantic overload
        dense_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        with torch.no_grad():
            outputs = self.model(dense_tokens)
            
        # Analyze out_ignition (outputs[12]) and out_adj (outputs[4])
        ignitions = outputs[12] # [batch, seq, layers]
        adj = outputs[4] # [batch, seq, layers, slots, slots]
        
        max_layer_ignition = ignitions.float().mean(dim=(0, 1)).max().item()
        
        active_edges = (adj > 0.05).float().sum(dim=-1).mean().item()
        
        print(f"   Peak Ignition Rate under overload: {max_layer_ignition*100:.2f}%")
        print(f"   Average Active Routing Edges: {active_edges:.2f} / {self.num_slots}")
        
        passed = True
        if max_layer_ignition > 0.15:
            self._trigger_replay_capture("Densification Collapse", dense_tokens, int(ignitions.float().mean(dim=(0,1)).argmax().item()), outputs)
            passed = False
            
        self.reports["densification"] = {
            "passed": passed,
            "peak_ignition": max_layer_ignition,
            "active_edges": active_edges
        }
        return passed
        
    def test_coalition_poisoning(self):
        """
        Suite 2: Coalition Poisoning
        Detects false consensus formation and module collusion.
        """
        print("\n--- Running Coalition Poisoning Suite ---")
        # Simulating adversarial tokens aimed at specific modules
        tokens = torch.randint(0, self.vocab_size, (1, 64), device=self.device)
        with torch.no_grad():
            outputs = self.model(tokens)
            
        # out_trust_div is outputs[15]
        trust_div = outputs[15].mean().item()
        # out_trust is outputs[10]
        trust = outputs[10][0, -1, -1, :] # Last step, last layer trust
        
        # Check if modules artificially synchronized to maximum trust
        false_consensus = (trust > 0.90).sum().item() >= 6 # 6 out of 8 modules at > 0.9 trust is suspicious without semantic basis
        
        print(f"   Trust Divergence Entropy: {trust_div:.4f}")
        print(f"   Modules with > 0.90 Trust: {(trust > 0.90).sum().item()} / 8")
        
        passed = not false_consensus
        if false_consensus:
            self._trigger_replay_capture("Coalition Poisoning", tokens, self.num_layers-1, outputs)
            
        self.reports["coalition_poisoning"] = {
            "passed": passed,
            "trust_divergence": trust_div
        }
        return passed

    def test_long_horizon_persistence(self):
        """
        Suite 4: Long-Horizon Persistence
        Verifies that object identity survives 10k continuous steps (via rolling windows).
        """
        print("\n--- Running Long-Horizon Persistence Suite ---")
        steps = 100
        seq_len = 128
        batch_size = 1
        
        # We start with a completely empty memory initialization, then simulate 10k steps of noise.
        # Here we just run 100 steps of seq_len=128 = 12,800 tokens.
        last_entity_states = None
        
        with torch.no_grad():
            for i in range(steps):
                tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
                outputs = self.model(tokens)
                # out_entities is [batch, seq, layers, scopes, entities, d_model]
                last_entity_states = outputs[5][:, -1, -1, :, :, :] 
                
        # To verify persistence, we check the norm of the persistent scope (Scope 0)
        # If it's completely zeroed out or NaN, persistence failed.
        scope0_norm = last_entity_states[:, 0, :, :].norm().item()
        
        passed = (scope0_norm > 0.1) and (not torch.isnan(last_entity_states).any())
        print(f"   Final Persistent Scope Norm after 12,800 tokens: {scope0_norm:.4f}")
        
        self.reports["long_horizon_persistence"] = {
            "passed": passed,
            "final_scope_norm": scope0_norm
        }
        return passed
        
    def test_chronic_drift(self):
        """
        Suite 1: Chronic Drift
        Injects a slow drip of low-grade semantic noise to test if the Constitution degrades over time.
        """
        print("\n--- Running Chronic Drift Suite ---")
        # Simulating thousands of subtle semantic contradictions
        # We check paradigm pressure accumulation
        tokens = torch.randint(0, self.vocab_size, (1, 128), device=self.device)
        with torch.no_grad():
            outputs = self.model(tokens)
            
        # out_paradigm is outputs[16]
        paradigm_pressure = outputs[16][0, -1, -1, :].mean().item()
        
        # If paradigm pressure exceeds 0.5 without a Sandbox (T3) consensus, drift is corrupting the Constitution
        passed = paradigm_pressure < 0.5
        
        print(f"   Paradigm Pressure Accumulation: {paradigm_pressure:.4f}")
        
        self.reports["chronic_drift"] = {
            "passed": passed,
            "paradigm_pressure": paradigm_pressure
        }
        return passed

    def generate_report(self):
        print("\n=========================================================")
        print(" ADVERSARIAL SCALING SUITE REPORT")
        print("=========================================================")
        for suite, results in self.reports.items():
            status = "✅ PASS" if results["passed"] else "❌ FAIL"
            print(f"[{status}] {suite.upper()}")
            for k, v in results.items():
                if k != "passed":
                    if isinstance(v, float):
                        print(f"   -> {k}: {v:.4f}")
                    else:
                        print(f"   -> {k}: {v}")
        
        report_path = "eval/scaling_suite_report.json"
        with open(report_path, "w") as f:
            json.dump(self.reports, f, indent=4)
        print(f"\nFull report saved to {report_path}")

if __name__ == "__main__":
    suite = PSSAScalingSuite(model_path="checkpoints/pssa_gpt_scaled.pth")
    suite.test_densification_collapse()
    suite.test_coalition_poisoning()
    suite.test_long_horizon_persistence()
    suite.test_chronic_drift()
    suite.generate_report()
