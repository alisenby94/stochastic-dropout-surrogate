"""
Simple Experiment: Compare Random Dropout vs Surrogate Dropout

Two separate training runs:
1. Baseline: Train with random masks only
2. Surrogate: Train with surrogate-predicted masks only

Track validation performance over time to see if surrogate helps or hurts.

Tests on both balanced and imbalanced datasets.
"""
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

from model import CNN, PathwayAwareSurrogate
from data import get_dataloaders


def train_without_dropout(model, train_loader, val_loader, device, num_epochs=100):
    """
    Train model WITHOUT dropout (control baseline).
    This shows what happens with no regularization.
    """
    print("\n" + "="*80)
    print("CONTROL: Training WITHOUT Dropout")
    print("="*80)
    
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    for epoch in range(num_epochs):
        import time
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass WITHOUT dropout (use normal forward, not forward_with_mask)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = F.cross_entropy(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        scheduler.step()
    
    return history


def create_imbalanced_dataset(train_loader, imbalance_ratio=0.3, num_classes=100):
    """
    Create an imbalanced version of the dataset.
    Randomly select half the classes and reduce their samples to imbalance_ratio.
    
    Args:
        train_loader: Original training dataloader
        imbalance_ratio: Fraction of samples to keep for minority classes (e.g., 0.3 = 30%)
        num_classes: Total number of classes
    
    Returns:
        New DataLoader with imbalanced class distribution
    """
    print(f"\nCreating imbalanced dataset (ratio={imbalance_ratio})...")
    
    # Get full dataset
    full_dataset = train_loader.dataset
    
    # Collect indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset):
        class_indices[label].append(idx)
    
    # Randomly select half the classes to be minority
    minority_classes = np.random.choice(num_classes, size=num_classes//2, replace=False)
    print(f"Minority classes (reduced to {imbalance_ratio*100:.0f}%): {sorted(minority_classes[:10])}... (50 total)")
    
    # Build new index list
    selected_indices = []
    class_counts = {}
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        
        if class_id in minority_classes:
            # Keep only imbalance_ratio of samples
            n_keep = max(1, int(len(indices) * imbalance_ratio))
            selected = np.random.choice(indices, size=n_keep, replace=False)
            selected_indices.extend(selected)
            class_counts[class_id] = n_keep
        else:
            # Keep all samples (majority class)
            selected_indices.extend(indices)
            class_counts[class_id] = len(indices)
    
    print(f"Original dataset size: {len(full_dataset)}")
    print(f"Imbalanced dataset size: {len(selected_indices)}")
    print(f"Majority class avg: {np.mean([class_counts[i] for i in range(num_classes) if i not in minority_classes]):.0f}")
    print(f"Minority class avg: {np.mean([class_counts[i] for i in minority_classes]):.0f}")
    
    # Create new dataset
    subset = Subset(full_dataset, selected_indices)
    new_loader = DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return new_loader, minority_classes


def train_with_random_masks(model, train_loader, val_loader, device, 
                            num_epochs=100, dropout_rate=0.5):
    """
    Train model using random dropout masks.
    Track validation performance over time.
    """
    print("\n" + "="*80)
    print("BASELINE: Training with Random Masks")
    print("="*80)
    
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    hidden_dim = 512
    k = int(hidden_dim * (1 - dropout_rate))  # Keep 50%
    
    for epoch in range(num_epochs):
        import time
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Get hidden activations
            with torch.no_grad():
                hidden = model.get_hidden(data)
            
            # Generate random mask (different for each sample in batch)
            batch_size = hidden.shape[0]
            mask = torch.zeros_like(hidden)
            for b in range(batch_size):
                perm = torch.randperm(hidden_dim, device=device)
                mask[b, perm[:k]] = 1.0
            
            # Forward pass with mask
            optimizer.zero_grad()
            output = model.forward_with_mask(data, mask)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                hidden = model.get_hidden(data)
                batch_size = hidden.shape[0]
                mask = torch.zeros_like(hidden)
                for b in range(batch_size):
                    perm = torch.randperm(hidden_dim, device=device)
                    mask[b, perm[:k]] = 1.0
                
                output = model.forward_with_mask(data, mask)
                loss = F.cross_entropy(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        scheduler.step()
    
    return history


def train_with_surrogate_masks(model, surrogate, train_loader, val_loader, device,
                               num_epochs=100, dropout_rate=0.5, 
                               phase1_epochs=10, phase1_batches=50):
    """
    Train model using surrogate-predicted masks.
    Track validation performance and mask statistics over time.
    """
    print("\n" + "="*80)
    print("SURROGATE: Training with Predicted Masks")
    print("="*80)
    
    model_optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    surrogate_optimizer = Adam(surrogate.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[50, 75], gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': [],
        'surrogate_trained': [],  # Track which epochs used trained surrogate
        
        # Mask statistics
        'mask_entropy': [],
        'mask_variance': [],
        'prob_mean': [],
        'prob_std': [],
        
        # Surrogate bootstrap accuracy
        'bootstrap_mask_accuracy': None  # Will be set after Phase 1
    }
    
    hidden_dim = 512
    k = int(hidden_dim * (1 - dropout_rate))  # Keep 50%
    
    # Phase 1: Bootstrap surrogate with random mask evaluation
    print("\n" + "="*60)
    print("PHASE 1: Bootstrap Surrogate")
    print("="*60)
    print(f"Evaluating random masks on {phase1_batches} batches × {phase1_epochs} epochs")
    
    activation_buffer = []
    
    samples = {'hidden': [], 'target_probs': [], 'weight': []}
    
    for epoch in range(phase1_epochs):
        pbar = tqdm(enumerate(train_loader), total=phase1_batches, 
                   desc=f'Phase 1 Epoch {epoch+1}/{phase1_epochs}')
        
        for batch_idx, (data, target) in pbar:
            if batch_idx >= phase1_batches:
                break
            
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                hidden = model.get_hidden(data)
            
            # Generate and evaluate 10 random masks
            num_masks = 10
            mask_results = []  # Store (mask, loss) pairs
            
            for _ in range(num_masks):
                batch_size = hidden.shape[0]
                mask = torch.zeros_like(hidden)
                for b in range(batch_size):
                    perm = torch.randperm(hidden_dim, device=device)
                    mask[b, perm[:k]] = 1.0
                
                # Evaluate this mask
                with torch.no_grad():
                    output = model.forward_with_mask(data, mask)
                    loss = F.cross_entropy(output, target)
                    mask_results.append((mask, loss.item()))
            
            # Sort by loss (most suppressive = highest loss = BEST for regularization)
            mask_results.sort(key=lambda x: x[1], reverse=True)
            
            # Best 3 masks (highest loss = most suppressive = GOOD regularization):
            # Neurons that were KEPT in these masks should have LOW dropout prob
            # (because keeping them creates good suppression)
            for mask, _ in mask_results[:3]:
                # mask=1 means kept (and caused high loss) → dropout_prob should be low (keep them)
                # mask=0 means dropped → dropout_prob should be high (drop them)
                dropout_probs = 1.0 - mask
                
                samples['hidden'].append(hidden.cpu())
                samples['target_probs'].append(dropout_probs.cpu())
                samples['weight'].append(1.0)
            
            # Worst 2 masks (lowest loss = least suppressive = BAD regularization):
            # Neurons that were KEPT in these masks should have HIGH dropout prob
            # (because keeping them made it too easy)
            for mask, _ in mask_results[-2:]:
                # mask=1 means kept (and caused low loss) → dropout_prob should be high (drop them)
                # mask=0 means dropped → dropout_prob should be low (keep them)
                dropout_probs = mask
                
                samples['hidden'].append(hidden.cpu())
                samples['target_probs'].append(dropout_probs.cpu())
                samples['weight'].append(0.5)  # Lower weight for negative examples
            
            pbar.set_postfix({'samples': len(samples['hidden'])})
    
    # Train surrogate on collected samples
    print(f"\nTraining surrogate on {len(samples['hidden'])} samples...")
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(
        torch.stack(samples['hidden']),
        torch.stack(samples['target_probs']),
        torch.tensor(samples['weight'])
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("\nTraining surrogate network...")
    avg_accuracy = 0.0  # Initialize for tracking
    for epoch in range(5):
        total_loss = 0
        total_accuracy = 0
        pbar = tqdm(loader, desc=f'Surrogate Epoch {epoch+1}/5')
        
        for hidden, target_probs, weight in pbar:
            hidden = hidden.to(device)
            target_probs = target_probs.to(device)
            weight = weight.to(device)
            
            surrogate_optimizer.zero_grad()
            
            # Predict dropout probabilities
            pred_probs = surrogate(hidden)
            
            # Handle 3D output (batch dimension from stored activations)
            if pred_probs.dim() == 3:
                pred_probs = pred_probs.mean(dim=1)  # [num_samples, 512]
            if target_probs.dim() == 3:
                target_probs = target_probs.mean(dim=1)  # [num_samples, 512]
            
            # MSE loss weighted by sample importance
            loss = F.mse_loss(pred_probs, target_probs, reduction='none')
            loss = (loss.mean(dim=1) * weight).mean()
            
            loss.backward()
            surrogate_optimizer.step()
            total_loss += loss.item()
            
            # Calculate binary mask accuracy
            # Convert probabilities to binary masks (threshold at 0.5)
            pred_mask = (pred_probs < 0.5).float()  # Low dropout prob = keep (1)
            target_mask = (target_probs < 0.5).float()  # Low dropout prob = keep (1)
            accuracy = (pred_mask == target_mask).float().mean().item()
            total_accuracy += accuracy
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': f'{total_accuracy / (pbar.n + 1)*100:.1f}%'
            })
        
        avg_loss = total_loss / len(loader)
        avg_accuracy = total_accuracy / len(loader)
        print(f"  Epoch {epoch+1}/5: Loss = {avg_loss:.4f}, Mask Accuracy = {avg_accuracy*100:.2f}%")
    
    # Save final bootstrap accuracy
    history['bootstrap_mask_accuracy'] = avg_accuracy
    
    print("\n✓ Surrogate trained!")
    print(f"Final mask accuracy: {avg_accuracy*100:.2f}% (how well surrogate replicates best suppression masks)")
    
    # Phase 2: Train model with surrogate masks
    print("\n" + "="*60)
    print("PHASE 2: Training with Surrogate Masks")
    print("="*60)
    
    for epoch in range(num_epochs):
        import time
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        epoch_probs = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Get hidden activations
            with torch.no_grad():
                hidden = model.get_hidden(data)
            
            # Generate surrogate mask
            with torch.no_grad():
                mask_probs = surrogate(hidden)
                epoch_probs.append(mask_probs.cpu().numpy())
            
            # Convert to top-k mask
            # mask_probs = dropout probability (high = drop, low = keep)
            # We want to KEEP neurons with LOW dropout probability
            _, top_indices = torch.topk(mask_probs, k, dim=1, largest=False)  # Get k SMALLEST probs
            mask = torch.zeros_like(mask_probs)
            mask.scatter_(1, top_indices, 1.0)  # mask=1 means keep these neurons
            
            # Forward pass with mask
            model_optimizer.zero_grad()
            output = model.forward_with_mask(data, mask)
            loss = F.cross_entropy(output, target)
            loss.backward()
            model_optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Buffer activations for correlation updates
            activation_buffer.append(hidden.detach().cpu())
            if len(activation_buffer) > 1000:
                activation_buffer.pop(0)
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Update correlation features periodically
        if hasattr(surrogate, 'update_correlation_features') and len(activation_buffer) > 100:
            recent_activations = torch.cat(activation_buffer[-100:], dim=0)
            surrogate.update_correlation_features(recent_activations)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                hidden = model.get_hidden(data)
                mask_probs = surrogate(hidden)
                # Keep neurons with LOWEST dropout probability
                _, top_indices = torch.topk(mask_probs, k, dim=1, largest=False)
                mask = torch.zeros_like(mask_probs)
                mask.scatter_(1, top_indices, 1.0)
                
                output = model.forward_with_mask(data, mask)
                loss = F.cross_entropy(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        epoch_time = time.time() - epoch_start
        
        # Compute mask statistics
        all_probs = np.concatenate(epoch_probs, axis=0)
        probs_avg = all_probs.mean(axis=0)
        
        eps = 1e-8
        entropy = -(probs_avg * np.log(probs_avg + eps) + 
                   (1 - probs_avg) * np.log(1 - probs_avg + eps)).mean()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        history['surrogate_trained'].append(True)
        history['mask_entropy'].append(float(entropy))
        history['mask_variance'].append(float(probs_avg.var()))
        history['prob_mean'].append(float(probs_avg.mean()))
        history['prob_std'].append(float(probs_avg.std()))
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        scheduler.step()
    
    return history


def compare_results(no_dropout_history, random_history, surrogate_history, save_dir='simple_comparison'):
    """
    Compare the two training runs and create visualizations.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw data
    with open(f'{save_dir}/random_history.json', 'w') as f:
        json.dump(random_history, f, indent=2)
    
    with open(f'{save_dir}/surrogate_history.json', 'w') as f:
        json.dump(surrogate_history, f, indent=2)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_control = range(1, len(no_dropout_history['val_loss']) + 1)
    epochs_random = range(1, len(random_history['val_loss']) + 1)
    epochs_surrogate = range(1, len(surrogate_history['val_loss']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs_control, no_dropout_history['train_loss'], 
                    label='No Dropout (Control)', marker='^', markersize=3, linewidth=2, color='gray')
    axes[0, 0].plot(epochs_random, random_history['train_loss'], 
                    label='Random Dropout', marker='o', markersize=3, linewidth=2, color='blue')
    axes[0, 0].plot(epochs_surrogate, surrogate_history['train_loss'], 
                    label='Surrogate Dropout', marker='s', markersize=3, linewidth=2, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0, 1].plot(epochs_control, no_dropout_history['val_loss'], 
                    label='No Dropout (Control)', marker='^', markersize=3, linewidth=2, color='gray')
    axes[0, 1].plot(epochs_random, random_history['val_loss'], 
                    label='Random Dropout', marker='o', markersize=3, linewidth=2, color='blue')
    axes[0, 1].plot(epochs_surrogate, surrogate_history['val_loss'], 
                    label='Surrogate Dropout', marker='s', markersize=3, linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mask Entropy (neuron usage diversity)
    if 'mask_entropy' in surrogate_history:
        axes[1, 0].plot(epochs_surrogate, surrogate_history['mask_entropy'], 
                        'b-', label='Surrogate Entropy', marker='s', markersize=3, linewidth=2)
        axes[1, 0].axhline(y=np.log(2), color='gray', linestyle='--', 
                          label='Max Entropy (uniform)', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mask Entropy (bits)')
        axes[1, 0].set_title('Neuron Selection Diversity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Dropout probability statistics (spread of neuron usage)
    if 'prob_mean' in surrogate_history and 'prob_std' in surrogate_history:
        axes[1, 1].plot(epochs_surrogate, surrogate_history['prob_mean'], 
                       'r-', label='Mean Dropout Prob', marker='s', markersize=3, linewidth=2)
        axes[1, 1].fill_between(epochs_surrogate,
                               np.array(surrogate_history['prob_mean']) - np.array(surrogate_history['prob_std']),
                               np.array(surrogate_history['prob_mean']) + np.array(surrogate_history['prob_std']),
                               alpha=0.3, label='±1 Std Dev')
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', 
                          label='Target (50%)', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Dropout Probability')
        axes[1, 1].set_title('Neuron Dropout Probability Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison.png', dpi=150)
    print(f"\n✓ Saved comparison plot: {save_dir}/comparison.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - LOSS & REGULARIZATION")
    print("="*80)
    
    print("\nRandom Masks:")
    print(f"  Final Train Loss: {random_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {random_history['val_loss'][-1]:.4f}")
    print(f"  Best Val Loss:    {min(random_history['val_loss']):.4f}")
    print(f"  Overfitting Gap:  {random_history['val_loss'][-1] - random_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Acc:    {random_history['val_acc'][-1]*100:.2f}%")
    print(f"  Avg Epoch Time:   {np.mean(random_history['epoch_times']):.1f}s")
    
    print("\nSurrogate Masks:")
    print(f"  Final Train Loss: {surrogate_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {surrogate_history['val_loss'][-1]:.4f}")
    print(f"  Best Val Loss:    {min(surrogate_history['val_loss']):.4f}")
    print(f"  Overfitting Gap:  {surrogate_history['val_loss'][-1] - surrogate_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Acc:    {surrogate_history['val_acc'][-1]*100:.2f}%")
    print(f"  Avg Epoch Time:   {np.mean(surrogate_history['epoch_times']):.1f}s")
    
    if 'mask_entropy' in surrogate_history:
        print(f"  Final Entropy:    {surrogate_history['mask_entropy'][-1]:.4f} bits")
        print(f"  Avg Entropy:      {np.mean(surrogate_history['mask_entropy']):.4f} bits")
        print(f"  Final Prob Mean:  {surrogate_history['prob_mean'][-1]:.4f}")
        print(f"  Final Prob Std:   {surrogate_history['prob_std'][-1]:.4f}")
    
    print("\nDifference (Surrogate - Random):")
    final_val_loss_diff = surrogate_history['val_loss'][-1] - random_history['val_loss'][-1]
    best_val_loss_diff = min(surrogate_history['val_loss']) - min(random_history['val_loss'])
    overfit_diff = (surrogate_history['val_loss'][-1] - surrogate_history['train_loss'][-1]) - \
                   (random_history['val_loss'][-1] - random_history['train_loss'][-1])
    
    print(f"  Final Val Loss:   {final_val_loss_diff:+.4f}")
    print(f"  Best Val Loss:    {best_val_loss_diff:+.4f}")
    print(f"  Overfitting Gap:  {overfit_diff:+.4f}")
    
    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)
    
    if final_val_loss_diff < -0.05:
        print("✓ Surrogate masks IMPROVED regularization (lower val loss)")
    elif final_val_loss_diff > 0.05:
        print("✗ Surrogate masks HURT regularization (higher val loss)")
    else:
        print("≈ Surrogate masks had NEGLIGIBLE effect on regularization")
    
    if overfit_diff < -0.05:
        print("✓ Surrogate masks REDUCED overfitting (smaller gap)")
    elif overfit_diff > 0.05:
        print("✗ Surrogate masks INCREASED overfitting (larger gap)")
    else:
        print("≈ Surrogate masks had NEGLIGIBLE effect on overfitting")


def run_experiment(train_loader, val_loader, device, num_epochs, 
                   experiment_name, save_dir, phase1_epochs=10, phase1_batches=50):
    """
    Run a complete experiment (no dropout, random dropout, surrogate dropout) on a given dataset.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        num_epochs: Number of training epochs
        experiment_name: Descriptive name (e.g., "balanced", "imbalanced_30pct")
        save_dir: Directory to save results
        phase1_epochs: Surrogate bootstrap epochs
        phase1_batches: Surrogate bootstrap batches
    
    Returns:
        (no_dropout_history, random_history, surrogate_history)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Output directory: {exp_dir}")
    print("="*80)
    
    # Experiment 0: No dropout (control)
    print("\n[1/3] Training WITHOUT dropout (control)...")
    model_control = CNN(num_classes=100).to(device)
    no_dropout_history = train_without_dropout(
        model_control, train_loader, val_loader, device, num_epochs=num_epochs
    )
    
    # Save checkpoint
    torch.save({
        'model': model_control.state_dict(),
        'history': no_dropout_history,
        'experiment': experiment_name,
        'timestamp': timestamp
    }, os.path.join(exp_dir, 'no_dropout_model.pth'))
    
    # Experiment 1: Random masks
    print("\n[2/3] Training with random masks...")
    model_random = CNN(num_classes=100).to(device)
    random_history = train_with_random_masks(
        model_random, train_loader, val_loader, device, num_epochs=num_epochs
    )
    
    # Save checkpoint
    torch.save({
        'model': model_random.state_dict(),
        'history': random_history,
        'experiment': experiment_name,
        'timestamp': timestamp
    }, os.path.join(exp_dir, 'random_model.pth'))
    
    # Experiment 2: Surrogate masks
    print("\n[3/3] Training with surrogate masks...")
    model_surrogate = CNN(num_classes=100).to(device)
    surrogate = PathwayAwareSurrogate(input_dim=512, hidden_dim=256).to(device)
    
    surrogate_history = train_with_surrogate_masks(
        model_surrogate, surrogate, train_loader, val_loader, device,
        num_epochs=num_epochs, phase1_epochs=phase1_epochs, phase1_batches=phase1_batches
    )
    
    # Save checkpoint
    torch.save({
        'model': model_surrogate.state_dict(),
        'surrogate': surrogate.state_dict(),
        'history': surrogate_history,
        'experiment': experiment_name,
        'timestamp': timestamp
    }, os.path.join(exp_dir, 'surrogate_model.pth'))
    
    # Compare results
    compare_results(no_dropout_history, random_history, surrogate_history, save_dir=exp_dir)
    
    print(f"\n✓ Experiment complete! Results saved to: {exp_dir}")
    
    return no_dropout_history, random_history, surrogate_history


def main():
    """Run full experimental pipeline on balanced and imbalanced datasets"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create base output directory
    save_dir = 'experiments'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
    num_epochs = 20  # Use 100 for full run
    
    # ============================================================================
    # EXPERIMENT 1: Balanced Dataset (original CIFAR-100)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1: BALANCED DATASET")
    print("="*80)
    
    balanced_control, balanced_random, balanced_surrogate = run_experiment(
        train_loader, val_loader, device, num_epochs,
        experiment_name="balanced",
        save_dir=save_dir
    )
    
    # ============================================================================
    # EXPERIMENT 2: Imbalanced Dataset (50% classes reduced to 30%)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 2: IMBALANCED DATASET")
    print("="*80)
    
    imbalanced_train_loader, minority_classes = create_imbalanced_dataset(
        train_loader, imbalance_ratio=0.3, num_classes=100
    )
    
    imbalanced_control, imbalanced_random, imbalanced_surrogate = run_experiment(
        imbalanced_train_loader, val_loader, device, num_epochs,
        experiment_name="imbalanced_30pct",
        save_dir=save_dir
    )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY - BALANCED vs IMBALANCED")
    print("="*80)
    
    print("\nBALANCED Dataset:")
    print(f"  No Dropout Val Loss:  {balanced_control['val_loss'][-1]:.4f}")
    print(f"  Random Val Loss:      {balanced_random['val_loss'][-1]:.4f}")
    print(f"  Surrogate Val Loss:   {balanced_surrogate['val_loss'][-1]:.4f}")
    print(f"  Random vs Control:    {balanced_random['val_loss'][-1] - balanced_control['val_loss'][-1]:+.4f}")
    print(f"  Surrogate vs Control: {balanced_surrogate['val_loss'][-1] - balanced_control['val_loss'][-1]:+.4f}")
    print(f"  Surrogate vs Random:  {balanced_surrogate['val_loss'][-1] - balanced_random['val_loss'][-1]:+.4f}")
    
    print("\nIMBALANCED Dataset:")
    print(f"  No Dropout Val Loss:  {imbalanced_control['val_loss'][-1]:.4f}")
    print(f"  Random Val Loss:      {imbalanced_random['val_loss'][-1]:.4f}")
    print(f"  Surrogate Val Loss:   {imbalanced_surrogate['val_loss'][-1]:.4f}")
    print(f"  Random vs Control:    {imbalanced_random['val_loss'][-1] - imbalanced_control['val_loss'][-1]:+.4f}")
    print(f"  Surrogate vs Control: {imbalanced_surrogate['val_loss'][-1] - imbalanced_control['val_loss'][-1]:+.4f}")
    print(f"  Surrogate vs Random:  {imbalanced_surrogate['val_loss'][-1] - imbalanced_random['val_loss'][-1]:+.4f}")
    
    print("\n" + "="*80)
    print(f"All results saved to: {save_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
