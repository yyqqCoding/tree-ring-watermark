"""
CoSDA Utility Functions

This module provides utility functions for:
1. Creating various image distortions for training
2. Evaluating inversion errors and metrics
3. Visualization and analysis tools
4. Integration helpers for Tree-Ring watermarking
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import io
from scipy.ndimage import median_filter


def create_distortions(
    image: torch.Tensor,
    distortion_types: List[str] = None,
    distortion_params: Dict = None
) -> Dict[str, torch.Tensor]:
    """
    Create various distortions for testing robustness.
    
    Args:
        image: Input image tensor (B, C, H, W) or (C, H, W)
        distortion_types: List of distortion types to apply
        distortion_params: Parameters for each distortion type
        
    Returns:
        Dictionary of distorted images
    """
    if distortion_types is None:
        distortion_types = ['jpeg', 'gaussian_noise', 'median_filter', 'crop_resize']
    
    if distortion_params is None:
        distortion_params = {
            'jpeg_qualities': [10, 30, 50, 70],
            'noise_sigmas': [0.01, 0.05, 0.1, 0.2],
            'filter_kernels': [3, 5, 7, 11],
            'crop_ratios': [0.7, 0.8, 0.9]
        }
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    distorted_images = {}
    
    for distortion_type in distortion_types:
        if distortion_type == 'jpeg':
            for quality in distortion_params.get('jpeg_qualities', [30]):
                key = f'jpeg_q{quality}'
                distorted_images[key] = apply_jpeg_compression(image, quality)
                
        elif distortion_type == 'gaussian_noise':
            for sigma in distortion_params.get('noise_sigmas', [0.1]):
                key = f'noise_s{sigma}'
                distorted_images[key] = apply_gaussian_noise(image, sigma)
                
        elif distortion_type == 'median_filter':
            for kernel in distortion_params.get('filter_kernels', [5]):
                key = f'filter_k{kernel}'
                distorted_images[key] = apply_median_filter(image, kernel)
                
        elif distortion_type == 'crop_resize':
            for ratio in distortion_params.get('crop_ratios', [0.8]):
                key = f'crop_r{ratio}'
                distorted_images[key] = apply_crop_resize(image, ratio)
    
    return distorted_images


def apply_jpeg_compression(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression to image tensor."""
    # Ensure image is in [0, 1] range
    image = torch.clamp(image, 0, 1)
    
    # Convert to PIL and apply JPEG compression
    compressed_images = []
    for i in range(image.shape[0]):
        img_pil = transforms.ToPILImage()(image[i])
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        compressed_tensor = transforms.ToTensor()(compressed_img)
        compressed_images.append(compressed_tensor)
    
    return torch.stack(compressed_images)


def apply_gaussian_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian noise to image tensor."""
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0, 1)


def apply_median_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply median filtering to image tensor."""
    filtered_images = []
    
    for i in range(image.shape[0]):
        img_np = image[i].permute(1, 2, 0).numpy()
        filtered_np = median_filter(img_np, size=(kernel_size, kernel_size, 1))
        filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).float()
        filtered_images.append(filtered_tensor)
    
    return torch.stack(filtered_images)


def apply_crop_resize(image: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    """Apply random crop and resize to image tensor."""
    _, _, h, w = image.shape
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    
    # Random crop
    top = torch.randint(0, h - crop_h + 1, (1,)).item()
    left = torch.randint(0, w - crop_w + 1, (1,)).item()
    cropped = image[:, :, top:top+crop_h, left:left+crop_w]
    
    # Resize back to original size
    return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)


def evaluate_inversion_error(
    original_latents: torch.Tensor,
    inverted_latents: torch.Tensor,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate inversion error using various metrics.
    
    Args:
        original_latents: Original latent tensors
        inverted_latents: Inverted latent tensors
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'cosine_similarity', 'psnr']
    
    results = {}
    
    # Flatten tensors for easier computation
    orig_flat = original_latents.flatten()
    inv_flat = inverted_latents.flatten()
    
    if 'mse' in metrics:
        mse = F.mse_loss(inverted_latents, original_latents)
        results['mse'] = mse.item()
    
    if 'mae' in metrics:
        mae = F.l1_loss(inverted_latents, original_latents)
        results['mae'] = mae.item()
    
    if 'cosine_similarity' in metrics:
        cos_sim = F.cosine_similarity(orig_flat, inv_flat, dim=0)
        results['cosine_similarity'] = cos_sim.item()
    
    if 'psnr' in metrics:
        mse_val = F.mse_loss(inverted_latents, original_latents)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_val))
        results['psnr'] = psnr.item()
    
    if 'ssim' in metrics:
        # Simplified SSIM calculation
        mu1 = original_latents.mean()
        mu2 = inverted_latents.mean()
        sigma1 = original_latents.var()
        sigma2 = inverted_latents.var()
        sigma12 = ((original_latents - mu1) * (inverted_latents - mu2)).mean()
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        results['ssim'] = ssim.item()
    
    return results


def visualize_inversion_comparison(
    original_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
    distorted_image: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize comparison between original and reconstructed images.
    
    Args:
        original_image: Original image tensor
        reconstructed_image: Reconstructed image tensor
        distorted_image: Optional distorted image tensor
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays
    def tensor_to_numpy(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    orig_np = tensor_to_numpy(original_image)
    recon_np = tensor_to_numpy(reconstructed_image)
    
    # Create figure
    if distorted_image is not None:
        dist_np = tensor_to_numpy(distorted_image)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(dist_np)
        axes[1].set_title('Distorted')
        axes[1].axis('off')
        
        axes[2].imshow(recon_np)
        axes[2].set_title('Reconstructed')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(recon_np)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_watermark_detection_metrics(
    watermark_scores_clean: List[float],
    watermark_scores_watermarked: List[float],
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute watermark detection metrics.
    
    Args:
        watermark_scores_clean: Scores for clean (non-watermarked) images
        watermark_scores_watermarked: Scores for watermarked images
        threshold: Detection threshold (if None, use optimal threshold)
        
    Returns:
        Dictionary of detection metrics
    """
    clean_scores = np.array(watermark_scores_clean)
    watermarked_scores = np.array(watermark_scores_watermarked)
    
    if threshold is None:
        # Find optimal threshold using ROC curve
        all_scores = np.concatenate([clean_scores, watermarked_scores])
        all_labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(watermarked_scores))])
        
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 100)
        best_acc = 0
        best_threshold = 0
        
        for t in thresholds:
            predictions = all_scores > t
            accuracy = (predictions == all_labels).mean()
            if accuracy > best_acc:
                best_acc = accuracy
                best_threshold = t
        
        threshold = best_threshold
    
    # Compute metrics
    clean_predictions = clean_scores > threshold
    watermarked_predictions = watermarked_scores > threshold
    
    # True Negative Rate (specificity)
    tnr = (clean_predictions == False).mean()
    
    # True Positive Rate (sensitivity/recall)
    tpr = (watermarked_predictions == True).mean()
    
    # False Positive Rate
    fpr = (clean_predictions == True).mean()
    
    # False Negative Rate
    fnr = (watermarked_predictions == False).mean()
    
    # Accuracy
    all_correct = np.concatenate([clean_predictions == False, watermarked_predictions == True])
    accuracy = all_correct.mean()
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'tpr': tpr,  # True Positive Rate
        'tnr': tnr,  # True Negative Rate
        'fpr': fpr,  # False Positive Rate
        'fnr': fnr,  # False Negative Rate
        'auc': compute_auc(clean_scores, watermarked_scores)
    }


def compute_auc(clean_scores: np.ndarray, watermarked_scores: np.ndarray) -> float:
    """Compute Area Under Curve (AUC) for watermark detection."""
    all_scores = np.concatenate([clean_scores, watermarked_scores])
    all_labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(watermarked_scores))])
    
    # Sort by scores
    sorted_indices = np.argsort(all_scores)
    sorted_labels = all_labels[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    tpr_values = []
    fpr_values = []
    
    for i in range(len(sorted_labels) + 1):
        if i == 0:
            threshold = sorted_labels[0] - 1
        elif i == len(sorted_labels):
            threshold = sorted_labels[-1] + 1
        else:
            threshold = all_scores[sorted_indices[i]]
        
        predictions = all_scores >= threshold
        tp = np.sum((predictions == True) & (all_labels == 1))
        fp = np.sum((predictions == True) & (all_labels == 0))
        tn = np.sum((predictions == False) & (all_labels == 0))
        fn = np.sum((predictions == False) & (all_labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_values, fpr_values)
    return abs(auc)  # Ensure positive AUC


def create_cosda_config(
    compensation_p: float = 0.8,
    enable_drift_alignment: bool = True,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    **kwargs
) -> Dict:
    """
    Create a configuration dictionary for CoSDA pipeline.
    
    Args:
        compensation_p: Compensation parameter for CoS sampling
        enable_drift_alignment: Whether to enable drift alignment
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        'compensation_p': compensation_p,
        'enable_drift_alignment': enable_drift_alignment,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'inversion_steps': kwargs.get('inversion_steps', 50),
        'inversion_guidance_scale': kwargs.get('inversion_guidance_scale', 1.0),
        'eta': kwargs.get('eta', 0.0),
    }
    
    # Add any additional parameters
    config.update(kwargs)
    
    return config
