"""
Training Script for Drift Alignment Network

This script trains the Drift Alignment Network to correct latent feature drift
caused by image distortions in the Tree-Ring watermarking pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CoSDA.drift_alignment import DriftAlignmentNetwork, DriftAlignmentTrainer, DistortionGenerator
from CoSDA.cosda_pipeline import CoSDAStableDiffusionPipeline
from CoSDA.utils import evaluate_inversion_error
from diffusers import StableDiffusionPipeline, DDIMScheduler
from optim_utils import get_watermarking_mask, inject_watermark


class DriftAlignmentDataGenerator:
    """Generate training data for Drift Alignment Network."""
    
    def __init__(
        self,
        pipeline: CoSDAStableDiffusionPipeline,
        num_samples: int = 1000,
        device: str = "cuda"
    ):
        self.pipeline = pipeline
        self.num_samples = num_samples
        self.device = device
        self.distortion_generator = DistortionGenerator()
        
    def generate_training_pairs(
        self,
        prompts: List[str],
        watermark_args: Dict,
        batch_size: int = 4
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate training pairs of (distorted_latent, original_latent).
        
        Args:
            prompts: List of text prompts for generation
            watermark_args: Watermarking configuration
            batch_size: Batch size for generation
            
        Returns:
            List of training pairs
        """
        training_pairs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.num_samples, batch_size), desc="Generating training data"):
                current_batch_size = min(batch_size, self.num_samples - i)
                
                # Select random prompts
                batch_prompts = np.random.choice(prompts, current_batch_size).tolist()
                
                for prompt in batch_prompts:
                    # Generate watermarked image
                    shape = (1, 4, 64, 64)
                    init_latents = torch.randn(shape, device=self.device)
                    
                    # Apply watermarking
                    watermarking_mask = get_watermarking_mask(init_latents, watermark_args, self.device)
                    gt_patch = torch.randn(shape, device=self.device)  # Simplified pattern
                    watermarked_latents = inject_watermark(init_latents, watermarking_mask, gt_patch, watermark_args)
                    
                    # Generate image
                    result = self.pipeline(
                        prompt=prompt,
                        latents=watermarked_latents,
                        num_inference_steps=20,  # Faster for training data generation
                        guidance_scale=7.5,
                        output_type="latent"
                    )
                    
                    # Decode to image
                    image = self.pipeline.vae.decode(result.images / self.pipeline.vae.config.scaling_factor).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    
                    # Apply random distortion
                    distorted_image = self.distortion_generator.apply_random_distortion(image)
                    
                    # Encode distorted image back to latent
                    distorted_latent = self.pipeline.vae.encode(distorted_image * 2 - 1).latent_dist.mode()
                    distorted_latent = distorted_latent * self.pipeline.vae.config.scaling_factor
                    
                    # Perform DDIM inversion on distorted latent
                    inverted_latent = self._ddim_inversion(distorted_latent, steps=10)
                    
                    # Store training pair
                    training_pairs.append({
                        'distorted_latent': inverted_latent.cpu(),
                        'target_latent': watermarked_latents.cpu(),
                        'prompt': prompt
                    })
        
        return training_pairs
    
    def _ddim_inversion(self, latent: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Perform simplified DDIM inversion."""
        # Get null embeddings
        null_embeddings = self.pipeline.get_null_text_embeddings(latent.shape[0])
        
        # Set timesteps
        self.pipeline.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        current_latent = latent
        
        for t in timesteps:
            # Predict noise
            noise_pred = self.pipeline.unet(current_latent, t, encoder_hidden_states=null_embeddings).sample
            
            # Inversion step (simplified)
            alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[t]
            next_timestep = t + self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
            alpha_prod_t_next = self.pipeline.scheduler.alphas_cumprod[next_timestep] if next_timestep < len(self.pipeline.scheduler.alphas_cumprod) else torch.tensor(0.0)
            
            # Predict x_0
            pred_original_sample = (current_latent - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            # Compute next sample
            pred_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
            current_latent = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction
        
        return current_latent


class DriftAlignmentDataset(torch.utils.data.Dataset):
    """Dataset for Drift Alignment Network training."""
    
    def __init__(self, training_pairs: List[Dict[str, torch.Tensor]]):
        self.training_pairs = training_pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        return {
            'distorted_latent': pair['distorted_latent'],
            'target_latent': pair['target_latent']
        }


def train_drift_alignment_network(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "./cosda_checkpoints",
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_training_samples: int = 5000,
    validation_split: float = 0.2,
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "cosda-drift-alignment"
):
    """
    Train the Drift Alignment Network.
    
    Args:
        model_id: Hugging Face model ID for Stable Diffusion
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        num_training_samples: Number of training samples to generate
        validation_split: Fraction of data for validation
        device: Device to train on
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
    """
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=wandb_project, config={
            'model_id': model_id,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_training_samples': num_training_samples
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print("Loading Stable Diffusion pipeline...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline = CoSDAStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    pipeline = pipeline.to(device)
    
    # Initialize Drift Alignment Network
    print("Initializing Drift Alignment Network...")
    da_network = DriftAlignmentNetwork(in_channels=4, hidden_channels=64)
    trainer = DriftAlignmentTrainer(
        network=da_network,
        device=device,
        learning_rate=learning_rate
    )
    
    # Generate training data
    print("Generating training data...")
    data_generator = DriftAlignmentDataGenerator(
        pipeline=pipeline,
        num_samples=num_training_samples,
        device=device
    )
    
    # Sample prompts for training
    sample_prompts = [
        "a beautiful landscape",
        "a portrait of a person",
        "a cat sitting on a table",
        "a modern building",
        "abstract art",
        "a flower in a garden",
        "a car on a road",
        "a bird flying in the sky",
        "a mountain view",
        "a beach scene"
    ]
    
    # Watermark configuration
    watermark_args = {
        'w_channel': 0,
        'w_radius': 10,
        'w_pattern': 'rand',
        'w_injection': 'complex',
        'w_measurement': 'complex'
    }
    
    training_pairs = data_generator.generate_training_pairs(
        prompts=sample_prompts,
        watermark_args=watermark_args,
        batch_size=4
    )
    
    # Split into train and validation
    split_idx = int(len(training_pairs) * (1 - validation_split))
    train_pairs = training_pairs[:split_idx]
    val_pairs = training_pairs[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = DriftAlignmentDataset(train_pairs)
    val_dataset = DriftAlignmentDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation
        val_loss = trainer.validate(val_loader)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'best_drift_alignment.pth')
            trainer.save_checkpoint(checkpoint_path, epoch, val_loss)
            print(f"Saved best checkpoint: {checkpoint_path}")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'drift_alignment_epoch_{epoch+1}.pth')
            trainer.save_checkpoint(checkpoint_path, epoch, val_loss)
    
    print("Training completed!")
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Drift Alignment Network for CoSDA")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="./cosda_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_training_samples", type=int, default=5000,
                        help="Number of training samples to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cosda-drift-alignment",
                        help="W&B project name")
    
    args = parser.parse_args()
    
    train_drift_alignment_network(
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_samples=args.num_training_samples,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    main()
