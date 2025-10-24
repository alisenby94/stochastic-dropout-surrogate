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
        """Quick validation on cached batches with proper mask handling"""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for data, target in self.val_cache:
                # If mask is 1D, expand to batch
                if mask.dim() == 1:
                    expanded_mask = mask.unsqueeze(0).expand(data.size(0), -1)
                else:
                    # If mask is 2D but doesn't match batch size, use first mask repeated
                    if mask.size(0) != data.size(0):
                        expanded_mask = mask[0].unsqueeze(0).expand(data.size(0), -1)
                    else:
                        expanded_mask = mask
                
                output = self.model.forward_with_mask(data, expanded_mask)
                loss = F.cross_entropy(output, target)
                losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses)
    
    def phase1_supervised(self, num_epochs=10, num_masks=10, dropout_rate=0.5, 
                     patience=3, min_delta=0.001):
        """Phase 1: Learn which masks lead to best improvements"""
        print("\n" + "="*60)
        print("PHASE 1: Supervised Bootstrap")
        print("="*60)
        
        samples = {'hidden': [], 'mask': [], 'improvement': []}
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            epoch_val_losses = []
            epoch_train_correct = 0
            epoch_train_total = 0
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                # Get hidden activations from current model state
                with torch.no_grad():
                    hidden = self.model.get_hidden(data)
                
                # Generate random masks per sample
                sample_masks = []
                for _ in range(num_masks):
                    batch_mask = torch.zeros(batch_size, hidden.shape[1], device=self.device)
                    for b in range(batch_size):
                        k = int(hidden.shape[1] * (1 - dropout_rate))
                        perm = torch.randperm(hidden.shape[1], device=self.device)
                        batch_mask[b, perm[:k]] = 1.0
                    sample_masks.append(batch_mask)
                
                # Save current model state
                batch_start_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                branch_results = []
                
                # Train each of the 10 branches and measure improvement
                for branch_idx, batch_mask in enumerate(sample_masks):
                    self.model.load_state_dict(batch_start_state)
                    
                    # Measure validation loss BEFORE training
                    val_loss_before = 0
                    self.model.eval()
                    with torch.no_grad():
                        for val_data, val_target in self.val_cache:
                            val_mask = batch_mask[0].unsqueeze(0).expand(val_data.size(0), -1)
                            val_output = self.model.forward_with_mask(val_data, val_mask)
                            val_loss_before += F.cross_entropy(val_output, val_target).item()
                    val_loss_before /= len(self.val_cache)
                    self.model.train()
                    
                    # Training step with this branch's masks
                    self.model_optimizer.zero_grad()
                    output = self.model.forward_with_mask(data, batch_mask)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.model_optimizer.step()
                    
                    # Measure validation loss AFTER training
                    val_loss_after = 0
                    self.model.eval()
                    with torch.no_grad():
                        for val_data, val_target in self.val_cache:
                            val_mask = batch_mask[0].unsqueeze(0).expand(val_data.size(0), -1)
                            val_output = self.model.forward_with_mask(val_data, val_mask)
                            val_loss_after += F.cross_entropy(val_output, val_target).item()
                    val_loss_after /= len(self.val_cache)
                    self.model.train()
                    
                    # Calculate improvement (positive = better, negative = worse)
                    improvement = val_loss_before - val_loss_after
                    
                    branch_results.append({
                        'branch_idx': branch_idx,
                        'masks': batch_mask,
                        'val_loss_after': val_loss_after,
                        'improvement': improvement,
                        'model_state': {k: v.clone() for k, v in self.model.state_dict().items()}
                    })
                
                # Sort by final validation loss (best model)
                sorted_branches = sorted(branch_results, key=lambda x: x['val_loss_after'])
                avg_val_loss = np.mean([r['val_loss_after'] for r in sorted_branches])
                epoch_val_losses.append(avg_val_loss)
                
                best_branch = sorted_branches[0]
                
                # Store ALL branches with their improvement scores
                # The surrogate will learn: high improvement = good mask, low/negative = bad mask
                for branch in branch_results:
                    for i in range(batch_size):
                        samples['hidden'].append(hidden[i].cpu())
                        samples['mask'].append(branch['masks'][i].cpu())
                        samples['improvement'].append(branch['improvement'])
                
                # Commit the winning branch's model state
                self.model.load_state_dict(best_branch['model_state'])
                
                # Track training accuracy with best branch
                with torch.no_grad():
                    train_output = self.model.forward_with_mask(data, best_branch['masks'])
                    train_pred = train_output.argmax(dim=1)
                    epoch_train_correct += (train_pred == target).sum().item()
                    epoch_train_total += target.size(0)
                
                # Update progress bar
                train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
                pbar.set_postfix({
                    'train_acc': f'{train_acc:.3f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'best_improvement': f'{best_branch["improvement"]:.4f}'
                })
            
            # Check early stopping at end of epoch (OUTSIDE batch loop)
            epoch_avg_val_loss = np.mean(epoch_val_losses)
            epoch_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
            print(f"  Epoch {epoch+1} - Train Acc: {epoch_train_acc:.3f}, Val Loss: {epoch_avg_val_loss:.4f}")
            
            if epoch_avg_val_loss < best_val_loss - min_delta:
                best_val_loss = epoch_avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Train surrogate to predict improvement scores
        print(f"\nTraining surrogate on {len(samples['hidden'])} samples...")
        print(f"Improvement range: [{min(samples['improvement']):.4f}, {max(samples['improvement']):.4f}]")
        
        dataset = TensorDataset(
            torch.stack(samples['hidden']),
            torch.stack(samples['mask']),
            torch.tensor(samples['improvement'], dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for epoch in range(10):
            total_loss = 0
            
            for hidden, mask, improvement in loader:
                hidden = hidden.to(self.device)
                mask = mask.to(self.device)
                improvement = improvement.to(self.device)
                
                self.surrogate_optimizer.zero_grad()
                
                # Surrogate predicts which neurons to keep (higher prob = keep)
                keep_probs = self.surrogate(hidden)
                
                # Key insight: For each neuron, we want to learn:
                # - If improvement was HIGH and neuron was KEPT (mask=1): predict HIGH prob
                # - If improvement was HIGH and neuron was DROPPED (mask=0): predict LOW prob
                # - If improvement was LOW and neuron was KEPT: predict LOW prob
                # - If improvement was LOW and neuron was DROPPED: predict HIGH prob
                
                # Normalize improvement to [0, 1] range (sigmoid-like)
                # Positive improvement -> target close to 1, negative -> target close to 0
                improvement_normalized = torch.sigmoid(improvement * 10.0)  # Scale for sensitivity
                
                # Target: For kept neurons (mask=1), match improvement
                #         For dropped neurons (mask=0), match (1-improvement)
                target = mask * improvement_normalized.unsqueeze(1) + \
                        (1 - mask) * (1 - improvement_normalized.unsqueeze(1))
                
                # Weight loss by absolute improvement (care more about high-impact examples)
                sample_weight = torch.abs(improvement).unsqueeze(1)
                sample_weight = sample_weight / (sample_weight.mean() + 1e-8)  # Normalize
                
                loss = F.binary_cross_entropy(keep_probs, target, weight=sample_weight)
                
                loss.backward()
                self.surrogate_optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/10: Loss = {avg_loss:.4f}")
    
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
                
                # Rank by validation loss
                sorted_results = sorted(results, key=lambda x: x['val_loss'])
                best = sorted_results[0]
                worst = sorted_results[-1]
                
                # Load best model
                self.model.load_state_dict(best['model_state'])
                
                # Track wins
                if best['source'] == 'surrogate':
                    epoch_surr_wins += 1
                    surrogate_wins += 1
                else:
                    epoch_rand_wins += 1
                    random_wins += 1
                
                # Calculate relative performance (normalize val losses to [0, 1])
                val_losses = [r['val_loss'] for r in sorted_results]
                val_min, val_max = min(val_losses), max(val_losses)
                val_range = val_max - val_min if val_max > val_min else 1.0
                
                # Update surrogate with reward/punishment
                surrogate_result = next(r for r in results if r['source'] == 'surrogate')
                
                # Calculate advantage: negative if surrogate is good, positive if bad
                # (we want to minimize validation loss)
                normalized_performance = (surrogate_result['val_loss'] - val_min) / val_range
                
                if best['source'] == 'surrogate':
                    # Surrogate won - reward it (label=1, strong positive signal)
                    advantage = 1.0 - normalized_performance  # Close to 1.0
                    self._update_surrogate(hidden, surrogate_result['mask'], label=1.0, advantage=advantage)
                    accum_grads += 1
                elif worst['source'] == 'surrogate':
                    # Surrogate lost badly - punish it (label=0, strong negative signal)
                    advantage = normalized_performance  # Close to 1.0
                    self._update_surrogate(hidden, surrogate_result['mask'], label=0.0, advantage=advantage)
                    accum_grads += 1
                else:
                    # Surrogate in middle - moderate update based on relative performance
                    if normalized_performance < 0.5:
                        # Better than average - weak reward
                        advantage = 0.5 - normalized_performance
                        self._update_surrogate(hidden, surrogate_result['mask'], label=1.0, advantage=advantage)
                        accum_grads += 1
                    else:
                        # Worse than average - weak punishment
                        advantage = normalized_performance - 0.5
                        self._update_surrogate(hidden, surrogate_result['mask'], label=0.0, advantage=advantage)
                        accum_grads += 1
                
                # Optimizer step
                if accum_grads >= grad_accum:
                    self.surrogate_optimizer.step()
                    self.surrogate_optimizer.zero_grad()
                    accum_grads = 0
                
                # Update progress
                total = epoch_surr_wins + epoch_rand_wins
                win_rate = epoch_surr_wins / total if total > 0 else 0
                
                # Find surrogate rank (avoid tensor comparison issues)
                surrogate_rank = None
                for idx, result in enumerate(sorted_results):
                    if result['source'] == 'surrogate':
                        surrogate_rank = idx + 1
                        break
                
                pbar.set_postfix({
                    'surr_wr': f'{win_rate:.1%}',
                    'winner': best['source'],
                    'surr_rank': f'{surrogate_rank}/5'
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
