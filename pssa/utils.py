import torch
import torch.nn as nn
import numpy as np

def measure_layer_variances(model, data_loader, device='cpu'):
    """
    Computes the variance of activations across all layers for the first batch of data.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, activations = model(x)
            variances = []
            for act in activations:
                act_flat = act.view(act.size(0), -1)
                # Compute feature-wise variance, then average
                feature_vars = torch.var(act_flat, dim=0)
                variances.append(torch.mean(feature_vars).item())
            return variances


def measure_singular_values(model):
    """
    Computes the singular values of the weight matrices in the model.
    """
    singular_vals = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Flatten to 2D matrix
            w = param.view(param.size(0), -1).detach().cpu()
            if w.size(0) > 1 and w.size(1) > 1:
                try:
                    # Run SVD
                    _, s, _ = torch.svd(w)
                    singular_vals[name] = s.tolist()
                except Exception:
                    # Fallback for numerical SVD issues
                    singular_vals[name] = []
            else:
                singular_vals[name] = []
    return singular_vals


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Evaluates a model's loss and accuracy on a given dataloader.
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu'):
    """
    Trains a model and returns history logs of training/validation loss and accuracy.
    """
    model.to(device)
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
    return history
