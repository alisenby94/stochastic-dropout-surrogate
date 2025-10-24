"""
CIFAR-100 data loading utilities.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(data_dir='./data', batch_size=100, val_split=0.1, num_workers=4):
    """Get CIFAR-100 train/val/test dataloaders"""
    
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    
    # Split train into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
