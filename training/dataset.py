import torch
from torch.utils.data import Dataset
import random

class MathAdditionDataset(Dataset):
    def __init__(self, num_samples=10000, max_digits=2):
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.data = []
        
        # Vocab: 0-9, '+', '=', PAD
        # 0-9 -> 0-9, '+' -> 10, '=' -> 11, PAD -> 12
        self.vocab_size = 13
        
        for _ in range(num_samples):
            a = random.randint(1, 10**max_digits - 1)
            b = random.randint(1, 10**max_digits - 1)
            c = a + b
            
            eq_str = f"{a}+{b}={c}"
            max_len = max_digits * 2 + 1 + 1 + max_digits + 2
            
            idx_list = []
            for char in eq_str:
                if char.isdigit():
                    idx_list.append(int(char))
                elif char == '+':
                    idx_list.append(10)
                elif char == '=':
                    idx_list.append(11)
                    
            while len(idx_list) < max_len:
                idx_list.append(12) # PAD
                
            inputs = torch.tensor(idx_list[:-1], dtype=torch.long)
            targets = torch.tensor(idx_list[1:], dtype=torch.long)
            
            self.data.append((inputs, targets))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
