"""
Tree-Ring Watermarking Integration with CoSDA

This module provides integration between the CoSDA optimization framework
and the Tree-Ring watermarking method, demonstrating how to use CoSDA
to improve watermark robustness.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CoSDA import CoSDAStableDiffusionPipeline, DriftAlignmentNetwork, create_distortions, evaluate_inversion_error
from optim_utils import get_watermarking_mask, inject_watermark, eval_watermark, transform_img


class CoSDATreeRingWatermarker:
    """
    Enhanced Tree-Ring watermarking with CoSDA optimization.
    
    This class integrates the CoSDA framework with Tree-Ring watermarking
    to improve robustness against various distortions.
    """
    
    def __init__(
        self,
        pipeline: CoSDAStableDiffusionPipeline,
        drift_alignment_network: Optional[DriftAlignmentNetwork] = None,
        compensation_p: float = 0.8,
        device: str = "cuda"
    ):
        """
        Initialize CoSDA Tree-Ring Watermarker.
        
        Args:
            pipeline: CoSDA-enhanced Stable Diffusion pipeline
            drift_alignment_network: Pre-trained drift alignment network
            compensation_p: Compensation parameter for CoS sampling
            device: Device to run on
        """
        self.pipeline = pipeline
        self.device = device
        self.compensation_p = compensation_p
        
        # Set up drift alignment if provided
        if drift_alignment_network is not None:
            self.pipeline.set_drift_alignment_network(drift_alignment_network)
        
        # Configure pipeline for watermarking
        self.pipeline.enable_compensation_sampling(compensation_p)
        
    def generate_watermarked_image(
        self,
        prompt: str,
        watermark_args: Dict,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        enable_compensation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate watermarked image using CoSDA-enhanced pipeline.
        
        Args:
            prompt: Text prompt for generation
            watermark_args: Watermarking arguments (w_channel, w_radius, etc.)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            generator: Random generator for reproducibility
            enable_compensation: Whether to use compensation sampling
            
        Returns:
            Dictionary containing generated images and latents
        """
        # Generate initial latents
        shape = (1, 4, height // 8, width // 8)
        init_latents = torch.randn(shape, generator=generator, device=self.device)
        
        # Get watermarking mask and pattern
        watermarking_mask = get_watermarking_mask(init_latents, watermark_args, self.device)
        
        # Create watermark pattern
        gt_patch = self._create_watermark_pattern(shape, watermark_args, generator)
        
        # Inject watermark into initial latents
        watermarked_latents = inject_watermark(init_latents, watermarking_mask, gt_patch, watermark_args)
        
        # Generate image with CoSDA pipeline
        result = self.pipeline(
            prompt=prompt,
            latents=watermarked_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            enable_compensation=enable_compensation,
            compensation_p=self.compensation_p,
            return_dict=True
        )
        
        return {
            'image': result.images[0],
            'init_latents': watermarked_latents,
            'watermarking_mask': watermarking_mask,
            'gt_patch': gt_patch,
            'compensation_metrics': result.compensation_metrics
        }
    
    def extract_watermark(
        self,
        image: torch.Tensor,
        watermarking_mask: torch.Tensor,
        gt_patch: torch.Tensor,
        watermark_args: Dict,
        num_inversion_steps: int = 50,
        apply_drift_alignment: bool = True
    ) -> Dict[str, float]:
        """
        Extract watermark from image using CoSDA-enhanced inversion.
        
        Args:
            image: Input image tensor
            watermarking_mask: Watermarking mask
            gt_patch: Ground truth watermark pattern
            watermark_args: Watermarking arguments
            num_inversion_steps: Number of DDIM inversion steps
            apply_drift_alignment: Whether to apply drift alignment
            
        Returns:
            Dictionary containing extraction metrics
        """
        # Perform DDIM inversion with CoSDA enhancements
        inversion_result = self.pipeline.ddim_inversion(
            image=image,
            num_inference_steps=num_inversion_steps,
            apply_drift_alignment=apply_drift_alignment,
            return_dict=True
        )
        
        inverted_latents = inversion_result['latents']
        inversion_metrics = inversion_result['metrics']
        
        # Evaluate watermark
        watermark_metric = eval_watermark(
            reversed_latents_no_w=torch.zeros_like(inverted_latents),  # Placeholder
            reversed_latents_w=inverted_latents,
            watermarking_mask=watermarking_mask,
            gt_patch=gt_patch,
            args=watermark_args
        )[1]  # Get watermarked metric
        
        return {
            'watermark_score': watermark_metric,
            'inversion_metrics': inversion_metrics,
            'inverted_latents': inverted_latents
        }
    
    def evaluate_robustness(
        self,
        original_image: torch.Tensor,
        watermarking_mask: torch.Tensor,
        gt_patch: torch.Tensor,
        watermark_args: Dict,
        distortion_types: List[str] = None,
        num_inversion_steps: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate watermark robustness against various distortions.
        
        Args:
            original_image: Original watermarked image
            watermarking_mask: Watermarking mask
            gt_patch: Ground truth watermark pattern
            watermark_args: Watermarking arguments
            distortion_types: Types of distortions to test
            num_inversion_steps: Number of inversion steps
            
        Returns:
            Dictionary of robustness metrics for each distortion
        """
        if distortion_types is None:
            distortion_types = ['jpeg', 'gaussian_noise', 'median_filter', 'crop_resize']
        
        # Create distorted versions
        distorted_images = create_distortions(
            original_image,
            distortion_types=distortion_types
        )
        
        results = {}
        
        # Test original image (no distortion)
        original_result = self.extract_watermark(
            original_image, watermarking_mask, gt_patch, watermark_args, num_inversion_steps
        )
        results['original'] = {
            'watermark_score': original_result['watermark_score'],
            'drift_correction_applied': original_result['inversion_metrics'].get('drift_correction_applied', False)
        }
        
        # Test each distorted version
        for distortion_name, distorted_image in distorted_images.items():
            try:
                distorted_result = self.extract_watermark(
                    distorted_image, watermarking_mask, gt_patch, watermark_args, num_inversion_steps
                )
                
                results[distortion_name] = {
                    'watermark_score': distorted_result['watermark_score'],
                    'drift_correction_applied': distorted_result['inversion_metrics'].get('drift_correction_applied', False),
                    'correction_strength': distorted_result['inversion_metrics'].get('correction_strength', 0.0)
                }
            except Exception as e:
                print(f"Error processing {distortion_name}: {e}")
                results[distortion_name] = {
                    'watermark_score': float('inf'),  # Indicate failure
                    'error': str(e)
                }
        
        return results
    
    def compare_with_baseline(
        self,
        prompt: str,
        watermark_args: Dict,
        distortion_types: List[str] = None,
        num_samples: int = 5,
        generator_seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare CoSDA performance with baseline Tree-Ring watermarking.
        
        Args:
            prompt: Text prompt for generation
            watermark_args: Watermarking arguments
            distortion_types: Types of distortions to test
            num_samples: Number of samples to generate and test
            generator_seed: Seed for reproducible results
            
        Returns:
            Comparison results between CoSDA and baseline
        """
        if distortion_types is None:
            distortion_types = ['jpeg', 'gaussian_noise', 'median_filter']
        
        cosda_results = []
        baseline_results = []
        
        for i in range(num_samples):
            # Set generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(generator_seed + i)
            
            # Generate with CoSDA
            self.pipeline.enable_compensation_sampling()
            cosda_gen = self.generate_watermarked_image(
                prompt, watermark_args, generator=generator, enable_compensation=True
            )
            
            # Generate with baseline (no compensation)
            self.pipeline.disable_compensation_sampling()
            baseline_gen = self.generate_watermarked_image(
                prompt, watermark_args, generator=generator, enable_compensation=False
            )
            
            # Test robustness for both
            cosda_rob = self.evaluate_robustness(
                transform_img(cosda_gen['image']).unsqueeze(0),
                cosda_gen['watermarking_mask'],
                cosda_gen['gt_patch'],
                watermark_args,
                distortion_types
            )
            
            baseline_rob = self.evaluate_robustness(
                transform_img(baseline_gen['image']).unsqueeze(0),
                baseline_gen['watermarking_mask'],
                baseline_gen['gt_patch'],
                watermark_args,
                distortion_types
            )
            
            cosda_results.append(cosda_rob)
            baseline_results.append(baseline_rob)
        
        # Aggregate results
        comparison = {}
        for distortion in ['original'] + list(distortion_types):
            cosda_scores = [r.get(distortion, {}).get('watermark_score', float('inf')) for r in cosda_results]
            baseline_scores = [r.get(distortion, {}).get('watermark_score', float('inf')) for r in baseline_results]
            
            # Filter out failed extractions
            cosda_scores = [s for s in cosda_scores if s != float('inf')]
            baseline_scores = [s for s in baseline_scores if s != float('inf')]
            
            if cosda_scores and baseline_scores:
                comparison[distortion] = {
                    'cosda_mean': np.mean(cosda_scores),
                    'cosda_std': np.std(cosda_scores),
                    'baseline_mean': np.mean(baseline_scores),
                    'baseline_std': np.std(baseline_scores),
                    'improvement': np.mean(cosda_scores) - np.mean(baseline_scores)
                }
        
        return comparison
    
    def _create_watermark_pattern(
        self,
        shape: Tuple[int, int, int, int],
        watermark_args: Dict,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Create watermark pattern based on watermark arguments."""
        # This is a simplified version - in practice, you'd use the actual
        # watermark pattern generation from the Tree-Ring implementation
        if watermark_args.get('w_pattern') == 'rand':
            return torch.randn(shape, generator=generator, device=self.device)
        elif watermark_args.get('w_pattern') == 'zeros':
            return torch.zeros(shape, device=self.device)
        else:
            # Default random pattern
            return torch.randn(shape, generator=generator, device=self.device)
