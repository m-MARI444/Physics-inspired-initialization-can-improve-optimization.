import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_spiral_data(n_samples=1500, n_classes=3, noise=0.2, seed=42):
    """
    Generates a 2D multi-class spiral dataset.
    """
    np.random.seed(seed)
    N = n_samples // n_classes
    D = 2  # 2D coordinates
    X = np.zeros((N * n_classes, D))
    y = np.zeros(N * n_classes, dtype='int64')
    
    for j in range(n_classes):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N)  # radius
        # spiral angle with noise
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
        
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def get_dataloaders(task_name='spiral', batch_size=128, n_samples=1500, n_classes=3, noise=0.2, seed=42):
    """
    Creates train and test DataLoaders for the specified task.
    """
    if task_name == 'spiral':
        X, y = get_spiral_data(n_samples=n_samples, n_classes=n_classes, noise=noise, seed=seed)
        
        # Split into train/test (80/20)
        indices = torch.randperm(len(X))
        split = int(0.8 * len(X))
        
        train_idx, test_idx = indices[:split], indices[split:]
        
        train_dataset = SyntheticDataset(X[train_idx], y[train_idx])
        test_dataset = SyntheticDataset(X[test_idx], y[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    elif task_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    else:
        raise ValueError(f"Unknown task name: {task_name}")
