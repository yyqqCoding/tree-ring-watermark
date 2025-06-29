"""
CoSDA Tree-Ring Watermarking Demo

This script demonstrates how to use the CoSDA optimization framework
with Tree-Ring watermarking to improve robustness against distortions.
"""

import torch
import numpy as np
import argparse
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CoSDA import CoSDAStableDiffusionPipeline, DriftAlignmentNetwork
from CoSDA.tree_ring_integration import CoSDATreeRingWatermarker
from CoSDA.utils import create_distortions, compute_watermark_detection_metrics
from diffusers import DDIMScheduler
from io_utils import transform_img


def load_drift_alignment_network(checkpoint_path: str, device: str = "cuda") -> DriftAlignmentNetwork:
    """Load pre-trained Drift Alignment Network."""
    network = DriftAlignmentNetwork(in_channels=4, hidden_channels=64)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Drift Alignment Network from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using untrained network")
    
    return network.to(device)


def demo_basic_usage(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    prompt: str = "a beautiful landscape with mountains and lakes",
    device: str = "cuda",
    output_dir: str = "./cosda_demo_outputs"
):
    """Demonstrate basic CoSDA usage with Tree-Ring watermarking."""
    
    print("=== CoSDA Tree-Ring Watermarking Demo ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print("1. Loading Stable Diffusion pipeline...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline = CoSDAStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipeline = pipeline.to(device)
    
    # Initialize watermarker
    print("2. Initializing CoSDA Tree-Ring Watermarker...")
    watermarker = CoSDATreeRingWatermarker(
        pipeline=pipeline,
        compensation_p=0.8,
        device=device
    )
    
    # Watermark configuration
    watermark_args = {
        'w_channel': 0,
        'w_radius': 10,
        'w_pattern': 'rand',
        'w_injection': 'complex',
        'w_measurement': 'complex'
    }
    
    # Generate watermarked image
    print("3. Generating watermarked image...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Generate with CoSDA (compensation enabled)
    cosda_result = watermarker.generate_watermarked_image(
        prompt=prompt,
        watermark_args=watermark_args,
        generator=generator,
        enable_compensation=True
    )
    
    # Generate baseline (compensation disabled)
    generator = torch.Generator(device=device).manual_seed(42)  # Same seed for fair comparison
    baseline_result = watermarker.generate_watermarked_image(
        prompt=prompt,
        watermark_args=watermark_args,
        generator=generator,
        enable_compensation=False
    )
    
    # Save generated images
    cosda_result['image'].save(os.path.join(output_dir, "cosda_watermarked.png"))
    baseline_result['image'].save(os.path.join(output_dir, "baseline_watermarked.png"))
    
    print(f"   CoSDA compensation metrics: {cosda_result['compensation_metrics']}")
    
    # Test watermark extraction on original images
    print("4. Testing watermark extraction on original images...")
    
    # Convert images to tensors
    cosda_img_tensor = transform_img(cosda_result['image']).unsqueeze(0).to(device)
    baseline_img_tensor = transform_img(baseline_result['image']).unsqueeze(0).to(device)
    
    # Extract watermarks
    cosda_extraction = watermarker.extract_watermark(
        cosda_img_tensor,
        cosda_result['watermarking_mask'],
        cosda_result['gt_patch'],
        watermark_args
    )
    
    baseline_extraction = watermarker.extract_watermark(
        baseline_img_tensor,
        baseline_result['watermarking_mask'],
        baseline_result['gt_patch'],
        watermark_args
    )
    
    print(f"   CoSDA watermark score: {cosda_extraction['watermark_score']:.4f}")
    print(f"   Baseline watermark score: {baseline_extraction['watermark_score']:.4f}")
    
    # Test robustness against distortions
    print("5. Testing robustness against distortions...")
    
    distortion_types = ['jpeg', 'gaussian_noise', 'median_filter']
    
    cosda_robustness = watermarker.evaluate_robustness(
        cosda_img_tensor,
        cosda_result['watermarking_mask'],
        cosda_result['gt_patch'],
        watermark_args,
        distortion_types=distortion_types
    )
    
    baseline_robustness = watermarker.evaluate_robustness(
        baseline_img_tensor,
        baseline_result['watermarking_mask'],
        baseline_result['gt_patch'],
        watermark_args,
        distortion_types=distortion_types
    )
    
    # Print robustness comparison
    print("\n   Robustness Comparison:")
    print("   " + "="*60)
    print(f"   {'Distortion':<20} {'CoSDA Score':<15} {'Baseline Score':<15} {'Improvement':<15}")
    print("   " + "-"*60)
    
    for distortion in ['original'] + [f'{dt}_q30' if 'jpeg' in dt else f'{dt}_s0.1' if 'noise' in dt else f'{dt}_k5' for dt in distortion_types]:
        if distortion in cosda_robustness and distortion in baseline_robustness:
            cosda_score = cosda_robustness[distortion]['watermark_score']
            baseline_score = baseline_robustness[distortion]['watermark_score']
            improvement = cosda_score - baseline_score
            
            print(f"   {distortion:<20} {cosda_score:<15.4f} {baseline_score:<15.4f} {improvement:<15.4f}")
    
    # Create visualization
    print("6. Creating visualization...")
    create_robustness_visualization(
        cosda_robustness,
        baseline_robustness,
        save_path=os.path.join(output_dir, "robustness_comparison.png")
    )
    
    print(f"\nDemo completed! Results saved to {output_dir}")
    
    return {
        'cosda_result': cosda_result,
        'baseline_result': baseline_result,
        'cosda_robustness': cosda_robustness,
        'baseline_robustness': baseline_robustness
    }


def demo_with_drift_alignment(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    drift_alignment_checkpoint: str = "./cosda_checkpoints/best_drift_alignment.pth",
    prompt: str = "a beautiful landscape with mountains and lakes",
    device: str = "cuda",
    output_dir: str = "./cosda_demo_outputs"
):
    """Demonstrate CoSDA with Drift Alignment Network."""
    
    print("=== CoSDA with Drift Alignment Demo ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print("1. Loading Stable Diffusion pipeline...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline = CoSDAStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipeline = pipeline.to(device)
    
    # Load Drift Alignment Network
    print("2. Loading Drift Alignment Network...")
    da_network = load_drift_alignment_network(drift_alignment_checkpoint, device)
    
    # Initialize watermarker with DA network
    print("3. Initializing CoSDA Tree-Ring Watermarker with Drift Alignment...")
    watermarker = CoSDATreeRingWatermarker(
        pipeline=pipeline,
        drift_alignment_network=da_network,
        compensation_p=0.8,
        device=device
    )
    
    # Watermark configuration
    watermark_args = {
        'w_channel': 0,
        'w_radius': 10,
        'w_pattern': 'rand',
        'w_injection': 'complex',
        'w_measurement': 'complex'
    }
    
    # Generate watermarked image
    print("4. Generating watermarked image...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    result = watermarker.generate_watermarked_image(
        prompt=prompt,
        watermark_args=watermark_args,
        generator=generator,
        enable_compensation=True
    )
    
    # Save generated image
    result['image'].save(os.path.join(output_dir, "cosda_da_watermarked.png"))
    
    # Test with heavy distortions
    print("5. Testing with heavy distortions...")
    
    img_tensor = transform_img(result['image']).unsqueeze(0).to(device)
    
    # Create heavy distortions
    heavy_distortions = create_distortions(
        img_tensor,
        distortion_types=['jpeg', 'gaussian_noise'],
        distortion_params={
            'jpeg_qualities': [5, 10],  # Very low quality
            'noise_sigmas': [0.2, 0.3]  # High noise
        }
    )
    
    print("   Testing extraction with and without Drift Alignment:")
    print("   " + "="*70)
    print(f"   {'Distortion':<20} {'With DA':<15} {'Without DA':<15} {'Improvement':<15}")
    print("   " + "-"*70)
    
    for distortion_name, distorted_img in heavy_distortions.items():
        # Extract with DA
        extraction_with_da = watermarker.extract_watermark(
            distorted_img,
            result['watermarking_mask'],
            result['gt_patch'],
            watermark_args,
            apply_drift_alignment=True
        )
        
        # Extract without DA
        extraction_without_da = watermarker.extract_watermark(
            distorted_img,
            result['watermarking_mask'],
            result['gt_patch'],
            watermark_args,
            apply_drift_alignment=False
        )
        
        score_with_da = extraction_with_da['watermark_score']
        score_without_da = extraction_without_da['watermark_score']
        improvement = score_with_da - score_without_da
        
        print(f"   {distortion_name:<20} {score_with_da:<15.4f} {score_without_da:<15.4f} {improvement:<15.4f}")
    
    print(f"\nDemo completed! Results saved to {output_dir}")


def create_robustness_visualization(
    cosda_results: dict,
    baseline_results: dict,
    save_path: str
):
    """Create visualization comparing robustness results."""
    
    # Extract scores for plotting
    distortions = []
    cosda_scores = []
    baseline_scores = []
    
    for distortion in cosda_results.keys():
        if distortion in baseline_results:
            distortions.append(distortion)
            cosda_scores.append(cosda_results[distortion]['watermark_score'])
            baseline_scores.append(baseline_results[distortion]['watermark_score'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(distortions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cosda_scores, width, label='CoSDA', alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_scores, width, label='Baseline', alpha=0.8)
    
    ax.set_xlabel('Distortion Type')
    ax.set_ylabel('Watermark Score')
    ax.set_title('Watermark Robustness: CoSDA vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(distortions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Robustness visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CoSDA Tree-Ring Watermarking Demo")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model ID")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and lakes",
                        help="Text prompt for generation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="./cosda_demo_outputs",
                        help="Output directory for results")
    parser.add_argument("--drift_alignment_checkpoint", type=str, default="./cosda_checkpoints/best_drift_alignment.pth",
                        help="Path to Drift Alignment Network checkpoint")
    parser.add_argument("--demo_type", type=str, choices=["basic", "drift_alignment", "both"], default="basic",
                        help="Type of demo to run")
    
    args = parser.parse_args()
    
    if args.demo_type in ["basic", "both"]:
        demo_basic_usage(
            model_id=args.model_id,
            prompt=args.prompt,
            device=args.device,
            output_dir=args.output_dir
        )
    
    if args.demo_type in ["drift_alignment", "both"]:
        demo_with_drift_alignment(
            model_id=args.model_id,
            drift_alignment_checkpoint=args.drift_alignment_checkpoint,
            prompt=args.prompt,
            device=args.device,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
