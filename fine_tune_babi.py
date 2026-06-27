import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Import model definition dynamically from our local training module
from kaggle_pssa_llm import PSSALanguageModel

# ==========================================
# 1. Synthetic bAbI Task 1 Generator
# ==========================================

PEOPLE = ["John", "Mary", "Sandra", "Daniel", "Daniela", "Robert", "James", "Patricia"]
LOCATIONS = ["office", "garden", "kitchen", "hallway", "bedroom", "bathroom", "park", "library"]

def generate_babi_sample():
    # Pick people and locations
    p1, p2, p3 = random.sample(PEOPLE, 3)
    l1, l2, l3 = random.sample(LOCATIONS, 3)
    
    # Generate stories
    story_templates = [
        f"{p1} went to the {l1}. {p2} moved to the {l2}. {p3} went back to the {l3}. Where is {p2}? ",
        f"{p1} travelled to the {l1}. {p2} went to the {l2}. Where is {p1}? ",
        f"{p1} moved to the {l1}. {p2} travelled to the {l2}. {p3} went to the {l3}. Where is {p3}? "
    ]
    
    idx = random.randint(0, 2)
    story = story_templates[idx]
    
    # Determine correct answer
    if idx == 0:
        answer = l2
    elif idx == 1:
        answer = l1
    else:
        answer = l3
        
    return story, answer

class BabiSyntheticDataset(Dataset):
    def __init__(self, tokenizer, num_samples=2000, max_len=64):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_len = max_len
        
        for _ in range(num_samples):
            story, answer = generate_babi_sample()
            # Tokenize story
            story_ids = tokenizer.encode(story)
            # Tokenize answer (we only want the first token of the answer)
            answer_id = tokenizer.encode(answer)[0]
            
            # Combine
            input_ids = story_ids + [answer_id]
            if len(input_ids) < max_len:
                # Pad to max_len
                input_ids = input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
            else:
                input_ids = input_ids[:max_len]
                
            # Create mask for loss (we only compute loss on the answer token!)
            loss_mask = [0] * (len(story_ids) - 1) + [1] + [0] * (max_len - len(story_ids))
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
                "targets": torch.tensor(input_ids[1:], dtype=torch.long),
                "mask": torch.tensor(loss_mask[:-1], dtype=torch.float32)
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 2. Evaluation Function
# ==========================================

def evaluate_babi_accuracy(model, tokenizer, device, num_tests=200, print_samples=10):
    model.eval()
    correct = 0
    samples_printed = 0
    with torch.no_grad():
        for _ in range(num_tests):
            story, answer = generate_babi_sample()
            inputs = tokenizer(story, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            
            logits, _, _ = model(input_ids)
            # Predict the token directly following the story context
            last_logits = logits[:, -1, :]
            pred_id = torch.argmax(last_logits, dim=-1).item()
            pred_word = tokenizer.decode([pred_id]).strip().lower()
            
            is_correct = (pred_word == answer.lower())
            if is_correct:
                correct += 1
                
            if samples_printed < print_samples:
                print(f"      [Sample] Story: {story.strip()}")
                print(f"               Target: {answer.lower():8s} | Predicted: {pred_word:8s} | Match: {is_correct}")
                samples_printed += 1
                
    accuracy = correct / num_tests
    return accuracy

# ==========================================
# 3. Main Training Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PSSA 100M model on bAbI Task 1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pssa_llm_kaggle.pth",
                        help="Path to the model weights checkpoint")
    parser.add_argument("--d_model", type=int, default=320,
                        help="Dimensionality of the model states")
    parser.add_argument("--num_slots", type=int, default=5,
                        help="Number of persistent slot memory states")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for fine-tuning")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning starting on device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        return
        
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    model = PSSALanguageModel(vocab_size=vocab_size, d_model=args.d_model, num_slots=args.num_slots)
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # 1. Baseline Evaluation (Before Fine-tuning)
    print("\nEvaluating baseline accuracy before fine-tuning...")
    initial_acc = evaluate_babi_accuracy(model, tokenizer, device, num_tests=100)
    print(f"🎯 Initial bAbI Task 1 Accuracy: {initial_acc * 100:.2f}%")
    
    # 2. Setup Dataloaders
    print("\nGenerating synthetic training samples...")
    train_dataset = BabiSyntheticDataset(tokenizer, num_samples=2500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 3. Fine-tuning Loop
    print("\n--- Starting bAbI Quick Fine-Tuning Campaign ---")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(input_ids) # [batch, seq_len, vocab_size]
            
            # Compute loss only on the answer token
            loss_elementwise = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="none")
            loss = torch.sum(loss_elementwise * mask.view(-1)) / (torch.sum(mask) + 1e-8)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Track training accuracy on target tokens
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                matches = (preds == targets).float() * mask
                correct_predictions += torch.sum(matches).item()
                total_predictions += torch.sum(mask).item()
                
        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        print(f"  > Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Training Acc: {train_acc * 100:.2f}%")
        
    # 4. Final Evaluation (After Fine-tuning)
    print("\nEvaluating final accuracy after fine-tuning...")
    final_acc = evaluate_babi_accuracy(model, tokenizer, device, num_tests=200)
    print(f"\n==============================================")
    print(f"🎯 Final bAbI Task 1 Accuracy: {final_acc * 100:.2f}%")
    print(f"==============================================")
    print(f"👉 Baseline Accuracy : {initial_acc * 100:.2f}%")
    print(f"👉 Post-Fine-Tune Accuracy: {final_acc * 100:.2f}%")
    print(f"==============================================")

if __name__ == "__main__":
    main()
