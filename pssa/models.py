import torch
import torch.nn as nn
import copy

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) classification/regression model.
    Returns both the logits and the intermediate activations for energy computation.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.out_layer = nn.Linear(in_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        """
        Forward pass.
        Returns:
            out (Tensor): Logits of shape (batch_size, output_dim)
            activations (list of Tensors): Activations at each layer, including input x.
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        activations = [x]
        curr = x
        for layer in self.layers:
            curr = self.activation_fn(layer(curr))
            activations.append(curr)
        out = self.out_layer(curr)
        activations.append(out)
        return out, activations


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for 28x28 images (like MNIST).
    Returns both the logits and flat/convolutional activations.
    """
    def __init__(self, in_channels=1, num_classes=10, activation_fn=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation_fn = activation_fn

    def forward(self, x):
        """
        Forward pass.
        Returns:
            out (Tensor): Logits of shape (batch_size, num_classes)
            activations (list of Tensors): Flat activations from each key stage of the network.
        """
        activations = []
        
        # Input layer (flattened for covariance analysis if needed, but we keep it flat here)
        activations.append(x.view(x.size(0), -1))
        
        # Conv 1
        c1 = self.activation_fn(self.conv1(x))
        # Flatten spatial dimensions for covariance analysis of features
        c1_flat = c1.permute(0, 2, 3, 1).reshape(c1.size(0), -1)
        activations.append(c1_flat)
        
        p1 = self.pool(c1)
        
        # Conv 2
        c2 = self.activation_fn(self.conv2(p1))
        c2_flat = c2.permute(0, 2, 3, 1).reshape(c2.size(0), -1)
        activations.append(c2_flat)
        
        p2 = self.pool(c2)
        
        # Flattened features
        flat = p2.view(p2.size(0), -1)
        activations.append(flat)
        
        # Fully Connected 1
        f1 = self.activation_fn(self.fc1(flat))
        activations.append(f1)
        
        # Fully Connected 2 (Output)
        out = self.fc2(f1)
        activations.append(out)
        
        return out, activations


def clone_model(model):
    """
    Creates a deep copy of a PyTorch model.
    """
    return copy.deepcopy(model)
