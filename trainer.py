"""
Tournament-based training with two phases:
1. Supervised bootstrap (10 epochs)
2. Online RL competition (90 epochs)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


class TournamentTrainer:
    
    def __init__(self, model, surrogate, train_loader, val_loader, device='cuda'):
        self.model = model
        self.surrogate = surrogate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.model.to(device)
        self.surrogate.to(device)
        
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.surrogate_optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.0001)
        
        # Cache validation batches for fast evaluation
        self.val_cache = []
        for i, (data, target) in enumerate(val_loader):
            if i >= 10:  # Cache first 10 batches
                break
            self.val_cache.append((data.to(device), target.to(device)))
    
    def generate_random_masks(self, hidden_dim, num_masks, dropout_rate=0.5):
        """Generate random masks with fixed sparsity"""
        k = int(hidden_dim * (1 - dropout_rate))
        masks = []
        
        for _ in range(num_masks):
            mask = torch.zeros(hidden_dim, device=self.device)
            perm = torch.randperm(hidden_dim, device=self.device)
            mask[perm[:k]] = 1.0
            masks.append(mask)
        
        return masks
    
    def fast_validate(self, mask):
        """Quick validation on cached batches"""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for data, target in self.val_cache:
                output = self.model.forward_with_mask(data, mask)
                loss = F.cross_entropy(output, target)
                losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses)
    
    def phase1_supervised(self, num_epochs=10, num_masks=10, k_best=3, k_worst=2, 
                         patience=3, min_delta=0.001):
        """Phase 1: Supervised bootstrap with random masks and early stopping"""
        print("\n" + "="*60)
        print("PHASE 1: Supervised Bootstrap")
        print("="*60)
        
        samples = {'hidden': [], 'mask': [], 'label': [], 'weight': []}
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get hidden activations
                with torch.no_grad():
                    hidden = self.model.get_hidden(data)
                
                # Generate random masks
                masks = self.generate_random_masks(hidden.shape[1], num_masks)
                
                # Evaluate each mask
                original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                results = []
                
                for mask in masks:
                    self.model.load_state_dict(original_state)
                    self.model_optimizer.zero_grad()
                    
                    # Training step
                    output = self.model.forward_with_mask(data, mask)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.model_optimizer.step()
                    
                    # Validation
                    val_loss = self.fast_validate(mask)
                    results.append({'mask': mask, 'val_loss': val_loss})
                
                # Collect top-k best and worst
                sorted_results = sorted(results, key=lambda x: x['val_loss'])
                
                # Only collect samples if they improve on validation
                # (avoid overfitting samples)
                avg_val_loss = np.mean([r['val_loss'] for r in sorted_results])
                if avg_val_loss < best_val_loss - min_delta:
                    for r in sorted_results[:k_best]:
                        samples['hidden'].append(hidden.cpu())
                        samples['mask'].append(r['mask'].cpu())
                        samples['label'].append(1.0)
                        samples['weight'].append(1.0)
                    
                    for r in sorted_results[-k_worst:]:
                        samples['hidden'].append(hidden.cpu())
                        samples['mask'].append(r['mask'].cpu())
                        samples['label'].append(0.0)
                        samples['weight'].append(0.5)
                    
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Reset model
                self.model.load_state_dict(original_state)
            
            # Check early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Train surrogate on collected samples
        print(f"\nTraining surrogate on {len(samples['hidden'])} samples...")
        dataset = TensorDataset(
            torch.stack(samples['hidden']),
            torch.stack(samples['mask']),
            torch.tensor(samples['label']),
            torch.tensor(samples['weight'])
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        for epoch in range(5):
            total_loss = 0
            for hidden, mask, label, weight in loader:
                hidden = hidden.to(self.device)
                mask = mask.to(self.device)
                label = label.to(self.device)
                weight = weight.to(self.device)
                
                self.surrogate_optimizer.zero_grad()
                
                probs = self.surrogate(hidden)
                loss = F.binary_cross_entropy(probs, mask, weight=weight.unsqueeze(1))
                
                loss.backward()
                self.surrogate_optimizer.step()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/5: Loss = {total_loss/len(loader):.4f}")
    
    def generate_competitors(self, hidden, dropout_rate=0.5):
        """Generate 1 surrogate + 4 random masks"""
        batch_size, hidden_dim = hidden.shape
        k = int(hidden_dim * (1 - dropout_rate))
        competitors = []
        
        # Surrogate mask
        mask_probs = self.surrogate(hidden)
        _, top_indices = torch.topk(mask_probs, k, dim=1)
        mask = torch.zeros_like(mask_probs)
        mask.scatter_(1, top_indices, 1.0)
        
        competitors.append({
            'mask': mask,
            'source': 'surrogate'
        })
        
        # 4 random masks
        for i in range(4):
            mask = torch.zeros(batch_size, hidden_dim, device=self.device)
            for b in range(batch_size):
                perm = torch.randperm(hidden_dim, device=self.device)
                mask[b, perm[:k]] = 1.0
            
            competitors.append({
                'mask': mask,
                'source': 'random'
            })
        
        return competitors
    
    def evaluate_competitors(self, data, target, competitors):
        """Evaluate all competitors in tournament"""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        results = []
        
        for comp in competitors:
            self.model.load_state_dict(original_state)
            self.model_optimizer.zero_grad()
            
            mask = comp['mask']
            
            # Measure before
            loss_before = self._compute_loss(data, target, mask)
            
            # Training step
            output = self.model.forward_with_mask(data, mask)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.model_optimizer.step()
            
            # Measure after
            loss_after = self._compute_loss(data, target, mask)
            
            # Validation
            val_loss = self.fast_validate(mask)
            
            results.append({
                'mask': mask,
                'source': comp['source'],
                'training_improvement': loss_before - loss_after,
                'val_loss': val_loss,
                'model_state': {k: v.clone() for k, v in self.model.state_dict().items()}
            })
        
        return results
    
    def _compute_loss(self, data, target, mask):
        """Compute loss with given mask"""
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward_with_mask(data, mask)
            loss = F.cross_entropy(output, target)
        self.model.train()
        return loss.item()
    
    def phase2_rl(self, num_epochs=90, grad_accum=5, dropout_rate=0.5,
                  patience=10, min_delta=0.01):
        """Phase 2: Online RL with 4v1 competition and early stopping"""
        print("\n" + "="*60)
        print("PHASE 2: Reinforcement Learning (4v1)")
        print("="*60)
        
        surrogate_wins = 0
        random_wins = 0
        best_win_rate = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            accum_grads = 0
            epoch_surr_wins = 0
            epoch_rand_wins = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get hidden activations
                with torch.no_grad():
                    hidden = self.model.get_hidden(data)
                
                # Generate competitors
                competitors = self.generate_competitors(hidden, dropout_rate)
                
                # Run tournament
                results = self.evaluate_competitors(data, target, competitors)
                
                # Find best
                best = min(results, key=lambda x: x['val_loss'])
                worst = max(results, key=lambda x: x['val_loss'])
                
                # Load best model
                self.model.load_state_dict(best['model_state'])
                
                # Track wins
                if best['source'] == 'surrogate':
                    epoch_surr_wins += 1
                    surrogate_wins += 1
                else:
                    epoch_rand_wins += 1
                    random_wins += 1
                
                # Update surrogate based on training improvements
                if best['source'] == 'surrogate' or worst['source'] == 'surrogate':
                    improvements = [r['training_improvement'] for r in results]
                    has_positive = any(imp > 0 for imp in improvements)
                    has_negative = any(imp < 0 for imp in improvements)
                    
                    # All positive: learn from highest positive
                    if has_positive and not has_negative:
                        if best['source'] == 'surrogate' and best['training_improvement'] > 0:
                            self._update_surrogate(hidden, best['mask'], label=1.0, 
                                                 advantage=best['training_improvement'])
                            accum_grads += 1
                    
                    # All negative: learn from lowest negative
                    elif has_negative and not has_positive:
                        if worst['source'] == 'surrogate' and worst['training_improvement'] < 0:
                            self._update_surrogate(hidden, worst['mask'], label=0.0,
                                                 advantage=worst['training_improvement'])
                            accum_grads += 1
                    
                    # Mixed: learn from both
                    else:
                        if best['source'] == 'surrogate' and best['training_improvement'] > 0:
                            self._update_surrogate(hidden, best['mask'], label=1.0,
                                                 advantage=best['training_improvement'])
                            accum_grads += 1
                        
                        if worst['source'] == 'surrogate' and worst['training_improvement'] < 0:
                            self._update_surrogate(hidden, worst['mask'], label=0.0,
                                                 advantage=worst['training_improvement'])
                            accum_grads += 1
                
                # Optimizer step
                if accum_grads >= grad_accum:
                    self.surrogate_optimizer.step()
                    self.surrogate_optimizer.zero_grad()
                    accum_grads = 0
                
                # Update progress
                total = epoch_surr_wins + epoch_rand_wins
                win_rate = epoch_surr_wins / total if total > 0 else 0
                pbar.set_postfix({
                    'surr_wr': f'{win_rate:.1%}',
                    'winner': best['source']
                })
            
            # Step at end of epoch
            if accum_grads > 0:
                self.surrogate_optimizer.step()
                self.surrogate_optimizer.zero_grad()
            
            # Epoch summary
            total = epoch_surr_wins + epoch_rand_wins
            win_rate = epoch_surr_wins / total if total > 0 else 0
            print(f"  Surrogate: {win_rate:.1%} ({epoch_surr_wins}/{total})")
            
            # Early stopping check
            if win_rate > best_win_rate + min_delta:
                best_win_rate = win_rate
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Final stats
        total = surrogate_wins + random_wins
        final_wr = surrogate_wins / total if total > 0 else 0
        print(f"\nFinal win rate: {final_wr:.1%} ({surrogate_wins}/{total})")
    
    def _update_surrogate(self, hidden, mask, label, advantage):
        """Update surrogate with REINFORCE"""
        probs = self.surrogate(hidden)
        target = torch.full_like(probs, label)
        
        loss = F.binary_cross_entropy(probs, mask) * abs(advantage)
        loss.backward()
