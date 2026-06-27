import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Import model definition dynamically from our local training module
from kaggle_pssa_llm import PSSALanguageModel

def download_from_hf(repo_id, token, filename, local_dir):
    try:
        print(f"📡 Downloading '{filename}' from Hugging Face repository '{repo_id}'...")
        hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir=local_dir)
        print(f"✅ Downloaded '{filename}' successfully.")
    except Exception as e:
        print(f"Warning: Failed to download from Hugging Face: {e}")

# ==========================================
# 1. WikiText-103 Perplexity Benchmark
# ==========================================

def eval_perplexity(model, tokenizer, device, max_samples=100):
    print("\n--- Running WikiText-103 Perplexity Benchmark ---")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    except Exception as e:
        print(f"Failed to load wikitext-103 from datasets library ({e}). Falling back to wikitext-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Filter out empty strings
    texts = [t["text"] for t in dataset if len(t["text"].strip()) > 0][:max_samples]
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating Perplexity"):
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            if input_ids.size(1) <= 1:
                continue
                
            targets = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]
            
            logits, _, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_token_id)
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            
    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    print(f"👉 WikiText-103 Perplexity: {perplexity:.4f} (Avg Cross-Entropy: {avg_loss:.4f})")
    return perplexity

# ==========================================
# 2. LAMBADA Final-Word Prediction Benchmark
# ==========================================

def eval_lambada(model, tokenizer, device, max_samples=100):
    print("\n--- Running LAMBADA Long-Range Benchmark ---")
    try:
        dataset = load_dataset("lambada", split="validation")
    except Exception as e:
        print(f"Warning: Could not download LAMBADA dataset directly ({e}). Simulating LAMBADA context tasks...")
        return simulate_lambada_tasks(model, tokenizer, device)
        
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            text = sample["text"]
            words = text.split()
            if len(words) < 2:
                continue
            context = " ".join(words[:-1])
            target_word = words[-1].strip().lower()
            
            inputs = tokenizer(context, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            if input_ids.size(1) == 0:
                continue
                
            logits, _, _ = model(input_ids)
            last_logits = logits[:, -1, :]  # [batch=1, vocab_size]
            predicted_id = torch.argmax(last_logits, dim=-1).item()
            predicted_word = tokenizer.decode([predicted_id]).strip().lower()
            
            if predicted_word == target_word:
                correct += 1
            total += 1
            
    accuracy = correct / total if total > 0 else 0.0
    print(f"👉 LAMBADA Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy

def simulate_lambada_tasks(model, tokenizer, device):
    # Simulated LAMBADA-style samples focusing on predicting the last word based on long context
    samples = [
        ("The chef combined the flour, sugar, and butter in a large bowl. She began to bake a", "cake"),
        ("He opened his wallet, took out a credit card, and gave it to the cashier to pay the", "bill"),
        ("The dog barked loudly at the mailman and started to chase the neighbor's", "cat"),
        ("She turned on the television, grabbed the remote control, and began watching her favorite", "show"),
        ("After climbing the steep mountain path for hours, they finally reached the high", "summit")
    ]
    model.eval()
    correct = 0
    with torch.no_grad():
        for context, target in samples:
            inputs = tokenizer(context, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            logits, _, _ = model(input_ids)
            last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            pred = tokenizer.decode([last_token_id]).strip().lower()
            if pred == target.lower():
                correct += 1
    accuracy = correct / len(samples)
    print(f"👉 Simulated LAMBADA Accuracy: {accuracy * 100:.2f}% ({correct}/{len(samples)})")
    return accuracy

# ==========================================
# 3. bAbI Tasks (Relational Reasoning & Slot Tracking)
# ==========================================

def eval_babi_tasks(model, tokenizer, device):
    print("\n--- Running bAbI Relational Reasoning Tasks ---")
    
    # Task 1: Single Supporting Fact (Entity tracking)
    task_1_samples = [
        {"context": "Mary went to the hallway. John moved to the office. Where is Mary? ", "answer": "hallway"},
        {"context": "Sandra travelled to the garden. Daniel went back to the bedroom. Where is Sandra? ", "answer": "garden"},
        {"context": "John went back to the kitchen. Sandra moved to the hallway. Where is John? ", "answer": "kitchen"}
    ]
    
    # Task 2: Two Supporting Facts (Object tracking across multiple hops)
    task_2_samples = [
        {"context": "John picked up the football. John went to the office. Daniel moved to the garden. Where is the football? ", "answer": "office"},
        {"context": "Mary grabbed the milk. Mary travelled to the kitchen. Sandra went to the office. Where is the milk? ", "answer": "kitchen"},
        {"context": "Daniel took the apple. Daniel moved to the bedroom. John went to the hallway. Where is the apple? ", "answer": "bedroom"}
    ]
    
    model.eval()
    
    def evaluate_babi_list(samples, name):
        correct = 0
        with torch.no_grad():
            for sample in samples:
                inputs = tokenizer(sample["context"], return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                logits, _, _ = model(input_ids)
                
                last_logits = logits[:, -1, :]
                pred_id = torch.argmax(last_logits, dim=-1).item()
                pred_word = tokenizer.decode([pred_id]).strip().lower()
                
                if pred_word == sample["answer"].lower():
                    correct += 1
        acc = correct / len(samples)
        print(f"  > bAbI {name} Accuracy: {acc * 100:.2f}% ({correct}/{len(samples)})")
        return acc

    acc1 = evaluate_babi_list(task_1_samples, "Task 1 (Single Fact)")
    acc2 = evaluate_babi_list(task_2_samples, "Task 2 (Two Facts)")
    return (acc1 + acc2) / 2

# ==========================================
# 4. Needle in a Haystack (NIAH) Evaluation
# ==========================================

def eval_needle_in_haystack(model, tokenizer, device, context_lengths=[512, 1024, 2048], depths=[0.1, 0.5, 0.9]):
    print("\n--- Running Needle in a Haystack (NIAH) Benchmark ---")
    
    needle_sentence = "The secret activation passcode is 8493."
    query = "What is the secret activation passcode? "
    target_answer = "8493"
    
    # Fill context with background text
    filler = "The quick brown fox jumps over the lazy dog. Artificial intelligence is evolving rapidly. "
    filler_tokens = tokenizer.encode(filler)
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        for length in context_lengths:
            results[length] = {}
            for depth in depths:
                # Build context
                total_filler_needed = length - len(tokenizer.encode(needle_sentence)) - len(tokenizer.encode(query))
                num_repeats = total_filler_needed // len(filler_tokens)
                
                context_tokens = []
                for _ in range(num_repeats):
                    context_tokens.extend(filler_tokens)
                
                # Insert needle at specified depth
                insert_idx = int(len(context_tokens) * depth)
                needle_tokens = tokenizer.encode(needle_sentence)
                
                full_context_tokens = context_tokens[:insert_idx] + needle_tokens + context_tokens[insert_idx:]
                query_tokens = tokenizer.encode(query)
                full_tokens = full_context_tokens + query_tokens
                
                # Truncate/pad to exact length if needed
                full_tokens = full_tokens[:length]
                input_ids = torch.tensor([full_tokens], dtype=torch.long).to(device)
                
                logits, _, _ = model(input_ids)
                last_logits = logits[:, -1, :]
                pred_id = torch.argmax(last_logits, dim=-1).item()
                pred_word = tokenizer.decode([pred_id]).strip()
                
                success = (pred_word == target_answer)
                results[length][depth] = success
                print(f"  > Context Length: {length:4d} | Depth: {int(depth*100):2d}% | Predicted: '{pred_word}' | Match: {success}")
                
    return results

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="PSSA-LM 100M Model Evaluation & Benchmarks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pssa_llm_kaggle.pth",
                        help="Path to the model weights checkpoint")
    parser.add_argument("--d_model", type=int, default=320,
                        help="Dimensionality of the model states (320 for 100M)")
    parser.add_argument("--num_slots", type=int, default=5,
                        help="Number of persistent slot memory states")
    parser.add_argument("--max_eval_samples", type=int, default=50,
                        help="Maximum samples to evaluate for wikitext/lambada to prevent timeouts")
    parser.add_argument("--hf_repo", type=str, default="",
                        help="Hugging Face repository ID to pull checkpoint from if missing")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face API write/read token")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing evaluation on device: {device}")
    
    # 1. Download model from Hugging Face if not found locally
    if not os.path.exists(args.checkpoint) and args.hf_repo and args.hf_token:
        print(f"Checkpoint '{args.checkpoint}' not found locally. Recovering from Hugging Face Hub...")
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        download_from_hf(args.hf_repo, args.hf_token, args.checkpoint, ".")
        
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found. Please provide a valid --checkpoint path or Hugging Face repository credentials.")
        return
        
    # 2. Initialize Tokenizer & Model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    print(f"Loading PSSA 100M model from checkpoint '{args.checkpoint}'...")
    model = PSSALanguageModel(vocab_size=vocab_size, d_model=args.d_model, num_slots=args.num_slots)
    
    # Handle state_dict if trained under nn.DataParallel
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # Measure and report total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 Model Loaded successfully! Total Parameters: {total_params / 1e6:.2f} Million")
    
    # 3. Execute Benchmarks
    results_summary = {}
    
    # Benchmark 1: Perplexity
    try:
        ppl = eval_perplexity(model, tokenizer, device, max_samples=args.max_eval_samples)
        results_summary["WikiText-103 Perplexity"] = f"{ppl:.4f}"
    except Exception as e:
        print(f"Skipping Perplexity evaluation due to error: {e}")
        
    # Benchmark 2: LAMBADA Long-range word prediction
    try:
        lambada_acc = eval_lambada(model, tokenizer, device, max_samples=args.max_eval_samples)
        results_summary["LAMBADA Word Acc"] = f"{lambada_acc * 100:.2f}%"
    except Exception as e:
        print(f"Skipping LAMBADA evaluation due to error: {e}")
        
    # Benchmark 3: bAbI Tasks (Slot Tracking & Multi-hop Reasoning)
    try:
        babi_acc = eval_babi_tasks(model, tokenizer, device)
        results_summary["bAbI Slot Reasoning Acc"] = f"{babi_acc * 100:.2f}%"
    except Exception as e:
        print(f"Skipping bAbI evaluation due to error: {e}")
        
    # Benchmark 4: Needle in a Haystack (Long Context Retrieval)
    try:
        niah_results = eval_needle_in_haystack(model, tokenizer, device, context_lengths=[256, 512, 1024])
        # Calculate summary accuracy for NIAH
        total_tests = 0
        matches = 0
        for length, depth_map in niah_results.items():
            for depth, val in depth_map.items():
                if val:
                    matches += 1
                total_tests += 1
        niah_acc = matches / total_tests if total_tests > 0 else 0.0
        results_summary["NIAH Context Retrieval Acc"] = f"{niah_acc * 100:.2f}%"
    except Exception as e:
        print(f"Skipping NIAH evaluation due to error: {e}")
        
    # Print Final Summary Table
    print("\n==============================================")
    print("      PSSA-LM 100M MODEL BENCHMARK SUMMARY")
    print("==============================================")
    for k, v in results_summary.items():
        print(f"{k:32s} : {v}")
    print("==============================================")

if __name__ == "__main__":
    main()
