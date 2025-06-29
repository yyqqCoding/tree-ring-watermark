"""
Comparison Script for Baseline Tree-Ring vs CoSDA-Enhanced Methods

This script helps analyze and compare the results from baseline Tree-Ring watermarking
and CoSDA-enhanced watermarking methods.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from typing import Dict, List, Tuple
import seaborn as sns


def load_wandb_results(run_path: str) -> Dict:
    """Load results from wandb run."""
    try:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        
        # Get summary metrics
        summary = run.summary
        
        # Get history for detailed analysis
        history = run.history()
        
        return {
            'summary': dict(summary),
            'history': history,
            'config': dict(run.config)
        }
    except Exception as e:
        print(f"Error loading wandb results: {e}")
        return None


def load_local_results(results_file: str) -> Dict:
    """Load results from local JSON file."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading local results: {e}")
        return None


def compare_clip_scores(baseline_results: Dict, cosda_results: Dict) -> Dict:
    """Compare CLIP scores between baseline and CoSDA methods."""
    
    comparison = {}
    
    # Extract CLIP scores
    baseline_clip_w = baseline_results.get('w_clip_scores', [])
    baseline_clip_no_w = baseline_results.get('no_w_clip_scores', [])
    cosda_clip_w = cosda_results.get('w_clip_scores', [])
    cosda_clip_no_w = cosda_results.get('no_w_clip_scores', [])
    
    if baseline_clip_w and cosda_clip_w:
        comparison['clip_scores'] = {
            'baseline': {
                'with_watermark': {
                    'mean': np.mean(baseline_clip_w),
                    'std': np.std(baseline_clip_w),
                    'scores': baseline_clip_w
                },
                'without_watermark': {
                    'mean': np.mean(baseline_clip_no_w),
                    'std': np.std(baseline_clip_no_w),
                    'scores': baseline_clip_no_w
                }
            },
            'cosda': {
                'with_watermark': {
                    'mean': np.mean(cosda_clip_w),
                    'std': np.std(cosda_clip_w),
                    'scores': cosda_clip_w
                },
                'without_watermark': {
                    'mean': np.mean(cosda_clip_no_w),
                    'std': np.std(cosda_clip_no_w),
                    'scores': cosda_clip_no_w
                }
            }
        }
        
        # Calculate improvements
        comparison['improvements'] = {
            'clip_score_with_watermark': np.mean(cosda_clip_w) - np.mean(baseline_clip_w),
            'clip_score_without_watermark': np.mean(cosda_clip_no_w) - np.mean(baseline_clip_no_w),
            'watermark_impact_baseline': np.mean(baseline_clip_w) - np.mean(baseline_clip_no_w),
            'watermark_impact_cosda': np.mean(cosda_clip_w) - np.mean(cosda_clip_no_w)
        }
    
    return comparison


def compare_watermark_detection(baseline_results: Dict, cosda_results: Dict) -> Dict:
    """Compare watermark detection performance."""
    
    comparison = {}
    
    # Extract watermark metrics
    baseline_w_metrics = baseline_results.get('w_metrics', [])
    baseline_no_w_metrics = baseline_results.get('no_w_metrics', [])
    cosda_w_metrics = cosda_results.get('w_metrics', [])
    cosda_no_w_metrics = cosda_results.get('no_w_metrics', [])
    
    if baseline_w_metrics and cosda_w_metrics:
        comparison['watermark_detection'] = {
            'baseline': {
                'with_watermark': {
                    'mean': np.mean(baseline_w_metrics),
                    'std': np.std(baseline_w_metrics),
                    'scores': baseline_w_metrics
                },
                'without_watermark': {
                    'mean': np.mean(baseline_no_w_metrics),
                    'std': np.std(baseline_no_w_metrics),
                    'scores': baseline_no_w_metrics
                }
            },
            'cosda': {
                'with_watermark': {
                    'mean': np.mean(cosda_w_metrics),
                    'std': np.std(cosda_w_metrics),
                    'scores': cosda_w_metrics
                },
                'without_watermark': {
                    'mean': np.mean(cosda_no_w_metrics),
                    'std': np.std(cosda_no_w_metrics),
                    'scores': cosda_no_w_metrics
                }
            }
        }
        
        # Calculate detection improvements
        comparison['detection_improvements'] = {
            'watermark_detection_improvement': np.mean(cosda_w_metrics) - np.mean(baseline_w_metrics),
            'false_positive_change': np.mean(cosda_no_w_metrics) - np.mean(baseline_no_w_metrics),
            'detection_gap_baseline': np.mean(baseline_w_metrics) - np.mean(baseline_no_w_metrics),
            'detection_gap_cosda': np.mean(cosda_w_metrics) - np.mean(cosda_no_w_metrics)
        }
    
    return comparison


def create_comparison_plots(comparison_data: Dict, output_dir: str = "./comparison_plots"):
    """Create visualization plots for comparison."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. CLIP Score Comparison
    if 'clip_scores' in comparison_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # With watermark comparison
        baseline_w = comparison_data['clip_scores']['baseline']['with_watermark']['scores']
        cosda_w = comparison_data['clip_scores']['cosda']['with_watermark']['scores']
        
        ax1.hist(baseline_w, alpha=0.7, label='Baseline', bins=30)
        ax1.hist(cosda_w, alpha=0.7, label='CoSDA', bins=30)
        ax1.set_xlabel('CLIP Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('CLIP Scores - With Watermark')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Without watermark comparison
        baseline_no_w = comparison_data['clip_scores']['baseline']['without_watermark']['scores']
        cosda_no_w = comparison_data['clip_scores']['cosda']['without_watermark']['scores']
        
        ax2.hist(baseline_no_w, alpha=0.7, label='Baseline', bins=30)
        ax2.hist(cosda_no_w, alpha=0.7, label='CoSDA', bins=30)
        ax2.set_xlabel('CLIP Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CLIP Scores - Without Watermark')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clip_score_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Watermark Detection Comparison
    if 'watermark_detection' in comparison_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # With watermark detection
        baseline_w = comparison_data['watermark_detection']['baseline']['with_watermark']['scores']
        cosda_w = comparison_data['watermark_detection']['cosda']['with_watermark']['scores']
        
        ax1.hist(baseline_w, alpha=0.7, label='Baseline', bins=30)
        ax1.hist(cosda_w, alpha=0.7, label='CoSDA', bins=30)
        ax1.set_xlabel('Watermark Detection Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Watermark Detection - With Watermark')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Without watermark detection (false positives)
        baseline_no_w = comparison_data['watermark_detection']['baseline']['without_watermark']['scores']
        cosda_no_w = comparison_data['watermark_detection']['cosda']['without_watermark']['scores']
        
        ax2.hist(baseline_no_w, alpha=0.7, label='Baseline', bins=30)
        ax2.hist(cosda_no_w, alpha=0.7, label='CoSDA', bins=30)
        ax2.set_xlabel('Watermark Detection Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Watermark Detection - Without Watermark')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'watermark_detection_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Summary Bar Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = []
    baseline_values = []
    cosda_values = []
    improvements = []
    
    if 'clip_scores' in comparison_data:
        metrics.extend(['CLIP Score\n(With Watermark)', 'CLIP Score\n(Without Watermark)'])
        baseline_values.extend([
            comparison_data['clip_scores']['baseline']['with_watermark']['mean'],
            comparison_data['clip_scores']['baseline']['without_watermark']['mean']
        ])
        cosda_values.extend([
            comparison_data['clip_scores']['cosda']['with_watermark']['mean'],
            comparison_data['clip_scores']['cosda']['without_watermark']['mean']
        ])
        improvements.extend([
            comparison_data['improvements']['clip_score_with_watermark'],
            comparison_data['improvements']['clip_score_without_watermark']
        ])
    
    if 'watermark_detection' in comparison_data:
        metrics.extend(['Watermark Detection\n(With Watermark)', 'Watermark Detection\n(Without Watermark)'])
        baseline_values.extend([
            comparison_data['watermark_detection']['baseline']['with_watermark']['mean'],
            comparison_data['watermark_detection']['baseline']['without_watermark']['mean']
        ])
        cosda_values.extend([
            comparison_data['watermark_detection']['cosda']['with_watermark']['mean'],
            comparison_data['watermark_detection']['cosda']['without_watermark']['mean']
        ])
        improvements.extend([
            comparison_data['detection_improvements']['watermark_detection_improvement'],
            comparison_data['detection_improvements']['false_positive_change']
        ])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, cosda_values, width, label='CoSDA', alpha=0.8)
    
    # Add improvement annotations
    for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        ax.annotate(f'+{improvement:.4f}' if improvement >= 0 else f'{improvement:.4f}',
                    xy=(i, height + 0.01),
                    ha='center', va='bottom',
                    fontweight='bold',
                    color='green' if improvement >= 0 else 'red')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs CoSDA Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")


def print_comparison_report(comparison_data: Dict):
    """Print detailed comparison report."""
    
    print("\n" + "="*80)
    print("BASELINE vs CoSDA COMPARISON REPORT")
    print("="*80)
    
    if 'clip_scores' in comparison_data:
        print("\n📊 CLIP SCORE ANALYSIS")
        print("-" * 40)
        
        baseline_w = comparison_data['clip_scores']['baseline']['with_watermark']
        cosda_w = comparison_data['clip_scores']['cosda']['with_watermark']
        baseline_no_w = comparison_data['clip_scores']['baseline']['without_watermark']
        cosda_no_w = comparison_data['clip_scores']['cosda']['without_watermark']
        
        print(f"With Watermark:")
        print(f"  Baseline:  {baseline_w['mean']:.4f} ± {baseline_w['std']:.4f}")
        print(f"  CoSDA:     {cosda_w['mean']:.4f} ± {cosda_w['std']:.4f}")
        print(f"  Improvement: {comparison_data['improvements']['clip_score_with_watermark']:.4f}")
        
        print(f"\nWithout Watermark:")
        print(f"  Baseline:  {baseline_no_w['mean']:.4f} ± {baseline_no_w['std']:.4f}")
        print(f"  CoSDA:     {cosda_no_w['mean']:.4f} ± {cosda_no_w['std']:.4f}")
        print(f"  Improvement: {comparison_data['improvements']['clip_score_without_watermark']:.4f}")
        
        print(f"\nWatermark Impact on CLIP Score:")
        print(f"  Baseline:  {comparison_data['improvements']['watermark_impact_baseline']:.4f}")
        print(f"  CoSDA:     {comparison_data['improvements']['watermark_impact_cosda']:.4f}")
    
    if 'watermark_detection' in comparison_data:
        print("\n🔍 WATERMARK DETECTION ANALYSIS")
        print("-" * 40)
        
        baseline_w = comparison_data['watermark_detection']['baseline']['with_watermark']
        cosda_w = comparison_data['watermark_detection']['cosda']['with_watermark']
        baseline_no_w = comparison_data['watermark_detection']['baseline']['without_watermark']
        cosda_no_w = comparison_data['watermark_detection']['cosda']['without_watermark']
        
        print(f"Watermark Detection (True Positives):")
        print(f"  Baseline:  {baseline_w['mean']:.4f} ± {baseline_w['std']:.4f}")
        print(f"  CoSDA:     {cosda_w['mean']:.4f} ± {cosda_w['std']:.4f}")
        print(f"  Improvement: {comparison_data['detection_improvements']['watermark_detection_improvement']:.4f}")
        
        print(f"\nFalse Positive Rate:")
        print(f"  Baseline:  {baseline_no_w['mean']:.4f} ± {baseline_no_w['std']:.4f}")
        print(f"  CoSDA:     {cosda_no_w['mean']:.4f} ± {cosda_no_w['std']:.4f}")
        print(f"  Change: {comparison_data['detection_improvements']['false_positive_change']:.4f}")
        
        print(f"\nDetection Gap (TP - FP):")
        print(f"  Baseline:  {comparison_data['detection_improvements']['detection_gap_baseline']:.4f}")
        print(f"  CoSDA:     {comparison_data['detection_improvements']['detection_gap_cosda']:.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline Tree-Ring vs CoSDA Results')
    parser.add_argument('--baseline_results', type=str, help='Path to baseline results JSON or wandb run path')
    parser.add_argument('--cosda_results', type=str, help='Path to CoSDA results JSON or wandb run path')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory for plots and reports')
    parser.add_argument('--use_wandb', action='store_true', help='Load results from wandb')
    
    args = parser.parse_args()
    
    # Load results
    if args.use_wandb:
        baseline_data = load_wandb_results(args.baseline_results)
        cosda_data = load_wandb_results(args.cosda_results)
    else:
        baseline_data = load_local_results(args.baseline_results)
        cosda_data = load_local_results(args.cosda_results)
    
    if baseline_data is None or cosda_data is None:
        print("Error: Could not load results data")
        return
    
    # Perform comparison
    clip_comparison = compare_clip_scores(baseline_data, cosda_data)
    detection_comparison = compare_watermark_detection(baseline_data, cosda_data)
    
    # Combine comparisons
    full_comparison = {**clip_comparison, **detection_comparison}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report
    print_comparison_report(full_comparison)
    
    # Create plots
    create_comparison_plots(full_comparison, args.output_dir)
    
    # Save comparison data
    with open(os.path.join(args.output_dir, 'comparison_data.json'), 'w') as f:
        json.dump(full_comparison, f, indent=2, default=str)
    
    print(f"\nComparison results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
