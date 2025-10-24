"""
Base CNN model for CIFAR-100.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Simple CNN with dropout layer for CIFAR-100"""
    
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """Standard forward pass"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_hidden(self, x):
        """Get hidden activations before final layer"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x
    
    def forward_with_mask(self, x, mask):
        """Forward pass with explicit dropout mask"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        
        # Apply mask
        x = x * mask
        
        x = self.fc2(x)
        return x


class SurrogateNetwork(nn.Module):
    """Predicts dropout masks from hidden activations"""
    
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, hidden):
        """Predict mask probabilities"""
        x = F.leaky_relu(self.fc1(hidden), negative_slope=0.01) # LeakyReLU: small gradient for negative inputs
        x = torch.sigmoid(self.fc2(x))                          # Sigmoid: outputs [0, 1] probabilities
        return x
