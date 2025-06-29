"""
Drift Alignment Network (DA) Implementation

This module implements the Drift Alignment Network to correct latent feature drift
caused by image distortions (JPEG compression, noise, filtering, etc.).

Network Architecture:
Single-Conv (Conv-BN-ReLU) → 3×ResBlock → Conv

Training objective:
L = ||Θ^DA(ž_V) - z_T||²

Where:
- ž_V: Distorted latent after V-step DDIM inversion
- z_T: Original watermark latent at timestep T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io


class ResidualBlock(nn.Module):
    """Residual block for Drift Alignment Network."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DriftAlignmentNetwork(nn.Module):
    """
    Drift Alignment Network to correct latent feature drift.
    
    Architecture: Single-Conv → 3×ResBlock → Conv
    Input: 4×64×64 latent features
    Output: 4×64×64 corrected latent features
    """
    
    def __init__(self, in_channels: int = 4, hidden_channels: int = 64):
        super().__init__()
        
        # Single-Conv layer
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3 Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(3)
        ])
        
        # Output layer
        self.output_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Drift Alignment Network.
        
        Args:
            x: Input latent tensor of shape (B, 4, H, W)
            
        Returns:
            Corrected latent tensor of shape (B, 4, H, W)
        """
        # Input convolution
        out = self.input_conv(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
            
        # Output convolution
        out = self.output_conv(out)
        
        return out


class DistortionGenerator:
    """Generate various image distortions for training data."""
    
    def __init__(self):
        self.jpeg_qualities = list(range(5, 51, 5))  # JPEG quality factors 5-50
        self.noise_sigmas = [0.01, 0.05, 0.1, 0.15, 0.2]  # Gaussian noise sigmas
        self.filter_kernels = [3, 5, 7, 9, 11, 15, 21]  # Median filter kernel sizes
        
    def apply_jpeg_compression(self, image: torch.Tensor, quality: int) -> torch.Tensor:
        """Apply JPEG compression distortion."""
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image.squeeze(0))
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        return transforms.ToTensor()(compressed_image).unsqueeze(0)
    
    def apply_gaussian_noise(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image + noise, 0, 1)
    
    def apply_median_filter(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filtering."""
        # Convert to numpy for median filtering
        image_np = image.squeeze(0).permute(1, 2, 0).numpy()
        from scipy.ndimage import median_filter
        
        filtered_np = median_filter(image_np, size=(kernel_size, kernel_size, 1))
        return torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
    
    def apply_random_crop_resize(self, image: torch.Tensor, crop_ratio: float = 0.8) -> torch.Tensor:
        """Apply random crop and resize."""
        _, _, h, w = image.shape
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        
        # Random crop
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        cropped = image[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize back
        return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
    
    def apply_random_distortion(self, image: torch.Tensor) -> torch.Tensor:
        """Apply a random distortion from available options."""
        distortion_type = np.random.choice(['jpeg', 'noise', 'filter', 'crop'])
        
        if distortion_type == 'jpeg':
            quality = np.random.choice(self.jpeg_qualities)
            return self.apply_jpeg_compression(image, quality)
        elif distortion_type == 'noise':
            sigma = np.random.choice(self.noise_sigmas)
            return self.apply_gaussian_noise(image, sigma)
        elif distortion_type == 'filter':
            kernel_size = np.random.choice(self.filter_kernels)
            return self.apply_median_filter(image, kernel_size)
        elif distortion_type == 'crop':
            crop_ratio = np.random.uniform(0.7, 0.9)
            return self.apply_random_crop_resize(image, crop_ratio)
        
        return image


class DriftAlignmentDataset(Dataset):
    """Dataset for training Drift Alignment Network."""
    
    def __init__(
        self,
        original_latents: List[torch.Tensor],
        watermark_latents: List[torch.Tensor],
        vae_encoder,
        vae_decoder,
        ddim_inverter,
        distortion_generator: DistortionGenerator,
        num_samples_per_latent: int = 5
    ):
        """
        Initialize dataset.
        
        Args:
            original_latents: List of original latent tensors z_T
            watermark_latents: List of watermark latent tensors
            vae_encoder: VAE encoder for image->latent conversion
            vae_decoder: VAE decoder for latent->image conversion
            ddim_inverter: DDIM inverter for latent inversion
            distortion_generator: Distortion generator
            num_samples_per_latent: Number of distorted samples per original latent
        """
        self.original_latents = original_latents
        self.watermark_latents = watermark_latents
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.ddim_inverter = ddim_inverter
        self.distortion_generator = distortion_generator
        self.num_samples_per_latent = num_samples_per_latent
        
    def __len__(self):
        return len(self.original_latents) * self.num_samples_per_latent
    
    def __getitem__(self, idx):
        # Get original latent
        latent_idx = idx // self.num_samples_per_latent
        z_T = self.original_latents[latent_idx]
        
        # Generate watermarked image
        with torch.no_grad():
            # Decode to image
            image = self.vae_decoder(z_T / 0.18215).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            
            # Apply distortion
            distorted_image = self.distortion_generator.apply_random_distortion(image)
            
            # Encode back to latent
            distorted_latent = self.vae_encoder(distorted_image * 2 - 1).latent_dist.mode() * 0.18215
            
            # Apply DDIM inversion
            inverted_latent = self.ddim_inverter.invert(distorted_latent, num_steps=10)
        
        return {
            'distorted_latent': inverted_latent,
            'target_latent': z_T
        }


class DriftAlignmentTrainer:
    """Trainer for Drift Alignment Network."""
    
    def __init__(
        self,
        network: DriftAlignmentNetwork,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.network = network.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            distorted_latents = batch['distorted_latent'].to(self.device)
            target_latents = batch['target_latent'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            corrected_latents = self.network(distorted_latents)
            
            # Compute loss
            loss = self.criterion(corrected_latents, target_latents)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                distorted_latents = batch['distorted_latent'].to(self.device)
                target_latents = batch['target_latent'].to(self.device)
                
                corrected_latents = self.network(distorted_latents)
                loss = self.criterion(corrected_latents, target_latents)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
