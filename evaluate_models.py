"""
Post-training evaluation with detailed per-class metrics.
Loads saved models and computes precision, recall, F1 for all classes.
"""

import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from data import get_dataloaders


def evaluate_model_detailed(model, data_loader, device, num_classes=100):
    """
    Evaluate model with detailed per-class metrics.
    
    Returns:
        dict with per-class and aggregated metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, 
        labels=list(range(num_classes)),
        zero_division=0
    )
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    
    # Overall accuracy
    accuracy = (all_preds == all_targets).mean()
    
    # Macro averages (treat all classes equally)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Micro averages (weight by support)
    micro_precision = (all_preds == all_targets).sum() / len(all_preds)
    micro_recall = micro_precision  # Same for multi-class
    micro_f1 = micro_precision
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        
        # Per-class metrics [100]
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        
        # Macro metrics (equal weight per class)
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        
        # Micro metrics (weight by samples)
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        
        # Confusion matrix
        'confusion_matrix': conf_matrix.tolist(),
    }


def analyze_imbalanced_performance(metrics, minority_classes):
    """
    Analyze performance on majority vs minority classes.
    
    Args:
        metrics: dict from evaluate_model_detailed
        minority_classes: list of class indices that were reduced
    
    Returns:
        dict with majority/minority breakdown
    """
    all_classes = set(range(100))
    majority_classes = list(all_classes - set(minority_classes))
    
    f1_scores = np.array(metrics['per_class_f1'])
    
    majority_f1 = f1_scores[majority_classes].mean()
    minority_f1 = f1_scores[list(minority_classes)].mean()
    
    return {
        'majority_classes': {
            'count': len(majority_classes),
            'avg_f1': float(majority_f1),
            'std_f1': float(f1_scores[majority_classes].std()),
        },
        'minority_classes': {
            'count': len(minority_classes),
            'avg_f1': float(minority_f1),
            'std_f1': float(f1_scores[list(minority_classes)].std()),
        },
        'minority_disadvantage': float(majority_f1 - minority_f1),
    }


def plot_per_class_comparison(results_dict, output_path, minority_classes=None):
    """
    Plot per-class F1 scores for all models.
    
    Args:
        results_dict: {model_name: metrics_dict}
        output_path: where to save plot
        minority_classes: list of minority class indices (for shading)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Comparison', fontsize=16)
    
    classes = list(range(100))
    
    # Plot 1: F1 scores
    ax = axes[0, 0]
    for name, metrics in results_dict.items():
        f1_scores = metrics['per_class_f1']
        ax.plot(classes, f1_scores, label=name, alpha=0.7, linewidth=1.5)
    
    if minority_classes:
        # Shade minority class regions
        for cls in minority_classes:
            ax.axvspan(cls - 0.5, cls + 0.5, alpha=0.1, color='red')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores (Red shading = Minority classes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Precision
    ax = axes[0, 1]
    for name, metrics in results_dict.items():
        precision = metrics['per_class_precision']
        ax.plot(classes, precision, label=name, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Precision')
    ax.set_title('Per-Class Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: Recall
    ax = axes[1, 0]
    for name, metrics in results_dict.items():
        recall = metrics['per_class_recall']
        ax.plot(classes, recall, label=name, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Support (sample count per class)
    ax = axes[1, 1]
    # Just show one model's support (they're all the same)
    first_model = list(results_dict.values())[0]
    support = first_model['per_class_support']
    colors = ['red' if i in (minority_classes or []) else 'blue' for i in classes]
    ax.bar(classes, support, color=colors, alpha=0.6, width=1.0)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Test Samples')
    ax.set_title('Test Set Class Distribution (Red = Minority)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-class comparison to {output_path}")


def plot_confusion_diff(conf_matrix_1, conf_matrix_2, model_name_1, model_name_2, output_path):
    """
    Plot difference between two confusion matrices.
    Shows where model 2 improves/worsens over model 1.
    """
    diff = np.array(conf_matrix_2) - np.array(conf_matrix_1)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(diff, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Prediction Count Difference'},
                xticklabels=5, yticklabels=5)
    plt.title(f'Confusion Matrix Difference ({model_name_2} - {model_name_1})\nBlue = {model_name_2} Better, Red = {model_name_1} Better')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion difference to {output_path}")


def evaluate_experiment(experiment_dir, device='cpu'):
    """
    Evaluate all models in an experiment directory.
    
    Args:
        experiment_dir: path to experiment (e.g., "experiments/20251126_180820_balanced")
    """
    exp_path = Path(experiment_dir)
    print(f"\n{'='*80}")
    print(f"EVALUATING: {exp_path.name}")
    print(f"{'='*80}\n")
    
    # Determine if imbalanced
    is_imbalanced = 'imbalanced' in exp_path.name
    minority_classes = None
    
    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=100)
    
    # Try to load minority classes info from surrogate checkpoint
    if is_imbalanced:
        surrogate_path = exp_path / 'surrogate_model.pth'
        if surrogate_path.exists():
            try:
                checkpoint = torch.load(surrogate_path, map_location='cpu')
                # Minority classes might be stored in the checkpoint
                # For now, we'll need to reconstruct from the experiment
                # This is a limitation - ideally we'd save this during training
                print("⚠ Note: Minority class indices not saved in checkpoint")
                print("   Cannot provide majority/minority breakdown")
            except Exception as e:
                print(f"⚠ Could not load surrogate checkpoint: {e}")
    
    # Evaluate each model
    results = {}
    model_files = {
        'No Dropout': exp_path / 'no_dropout_model.pth',
        'Random Dropout': exp_path / 'random_model.pth',
        'Surrogate Dropout': exp_path / 'surrogate_model.pth',
    }
    
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            print(f"⚠ {model_name} not found at {model_path}")
            continue
        
        print(f"Evaluating {model_name}...")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model = CNN(num_classes=100).to(device)
        model.load_state_dict(checkpoint['model'])
        
        # Evaluate
        metrics = evaluate_model_detailed(model, test_loader, device)
        results[model_name] = metrics
        
        # Print summary
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Micro F1:     {metrics['micro_f1']:.4f}")
        print(f"  Macro Prec:   {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        
        if is_imbalanced and minority_classes:
            imb_analysis = analyze_imbalanced_performance(metrics, minority_classes)
            print(f"  Majority F1:  {imb_analysis['majority_classes']['avg_f1']:.4f}")
            print(f"  Minority F1:  {imb_analysis['minority_classes']['avg_f1']:.4f}")
            print(f"  Disadvantage: {imb_analysis['minority_disadvantage']:.4f}")
        print()
    
    # Save detailed results
    results_path = exp_path / 'detailed_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed metrics to {results_path}\n")
    
    # Create visualizations
    if results:
        plot_per_class_comparison(
            results, 
            exp_path / 'per_class_comparison.png',
            minority_classes=minority_classes
        )
    
    # Confusion matrix difference
    if 'Random Dropout' in results and 'Surrogate Dropout' in results:
        plot_confusion_diff(
            results['Random Dropout']['confusion_matrix'],
            results['Surrogate Dropout']['confusion_matrix'],
            'Random Dropout',
            'Surrogate Dropout',
            output_path=exp_path / 'confusion_diff_surrogate_vs_random.png'
        )
    
    if 'No Dropout' in results and 'Surrogate Dropout' in results:
        plot_confusion_diff(
            results['No Dropout']['confusion_matrix'],
            results['Surrogate Dropout']['confusion_matrix'],
            'No Dropout',
            'Surrogate Dropout',
            output_path=exp_path / 'confusion_diff_surrogate_vs_control.png'
        )
    
    return results


def print_summary_table(all_results):
    """Print a nice summary table of all experiments."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    header = f"{'Experiment':<40} {'Model':<20} {'Acc':<8} {'Macro F1':<10} {'Micro F1':<10}"
    print(header)
    print("-" * len(header))
    
    for exp_name, results in sorted(all_results.items()):
        first = True
        for model_name, metrics in sorted(results.items()):
            exp_display = exp_name if first else ""
            print(f"{exp_display:<40} {model_name:<20} "
                  f"{metrics['accuracy']:<8.4f} {metrics['macro_f1']:<10.4f} {metrics['micro_f1']:<10.4f}")
            first = False
        print()


def main():
    """Evaluate all experiments in the experiments directory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    experiments_dir = Path('experiments')
    if not experiments_dir.exists():
        print("No experiments directory found!")
        return
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    
    if not exp_dirs:
        print("No experiments found!")
        return
    
    print(f"Found {len(exp_dirs)} experiments to evaluate\n")
    
    # Evaluate each experiment
    all_results = {}
    for exp_dir in exp_dirs:
        try:
            results = evaluate_experiment(exp_dir, device)
            all_results[exp_dir.name] = results
        except Exception as e:
            print(f"✗ Error evaluating {exp_dir.name}: {e}\n")
            continue
    
    # Print final summary
    if all_results:
        print_summary_table(all_results)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Check individual experiment directories for detailed plots.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
