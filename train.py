"""
Main training script for tournament-based dropout.
"""
import torch
from model import CNN, SurrogateNetwork
from data import get_dataloaders
from trainer import TournamentTrainer


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Create models
    model = CNN(num_classes=100)
    surrogate = SurrogateNetwork(input_dim=512, hidden_dim=128)
    
    # Create trainer
    trainer = TournamentTrainer(model, surrogate, train_loader, val_loader, device)
    
    # Phase 1: Supervised bootstrap - learn from all branches
    trainer.phase1_supervised(num_epochs=10, num_masks=10, dropout_rate=0.5)
    
    # Phase 2: Online RL
    trainer.phase2_rl(num_epochs=90, grad_accum=5, dropout_rate=0.5)
    
    # Save models
    torch.save({
        'model': model.state_dict(),
        'surrogate': surrogate.state_dict()
    }, 'checkpoint.pth')
    
    print("\nTraining complete! Saved to checkpoint.pth")


if __name__ == '__main__':
    main()
