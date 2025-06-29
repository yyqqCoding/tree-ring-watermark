"""
Compensation Sampling (CoS) Implementation

This module implements the compensation sampling mechanism to reduce condition mismatch
errors between forward sampling (with text condition C and guidance scale w > 1) and 
reverse inversion (with null condition ∅ and w = 1).

Key Formula:
x_{t-1} = γ_t * x_t + φ_t * [p * ε_θ(x_t,t,C,w) + (1-p) * ε_θ(x̄_{t-1},t,∅)]

Where:
- x̄_{t-1} = γ_t * x_t + φ_t * ε_θ(x_t,t,C,w) (temporary latent)
- p: compensation parameter (default 0.8)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List
import numpy as np


class CompensationSampler:
    """
    Implements Compensation Sampling for DDIM to reduce condition mismatch errors.
    """
    
    def __init__(self, p: float = 0.8):
        """
        Initialize CompensationSampler.
        
        Args:
            p (float): Compensation parameter controlling the weight between 
                      conditional and null condition components. Range [0, 1].
                      Smaller p enhances inversion robustness but may reduce image quality.
        """
        self.p = p
        
    def compute_noise_prediction(
        self,
        model,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float = 7.5,
        null_embeddings: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Compute noise prediction with classifier-free guidance.
        
        Args:
            model: UNet model for noise prediction
            latents: Current latent tensor
            timestep: Current timestep
            text_embeddings: Text condition embeddings
            guidance_scale: Guidance scale w
            null_embeddings: Null condition embeddings (empty prompt)
            
        Returns:
            Tuple of (conditional_noise_pred, unconditional_noise_pred)
        """
        # Prepare model input
        if null_embeddings is None:
            # Create null embeddings (empty prompt)
            batch_size = text_embeddings.shape[0]
            null_embeddings = torch.zeros_like(text_embeddings)
        
        # Concatenate for batch processing
        latent_model_input = torch.cat([latents] * 2)
        embeddings_input = torch.cat([null_embeddings, text_embeddings])
        
        # Predict noise
        noise_pred = model(latent_model_input, timestep, encoder_hidden_states=embeddings_input).sample
        
        # Split predictions
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Apply classifier-free guidance
        noise_pred_cond = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred_cond, noise_pred_uncond
    
    def compensation_step(
        self,
        model,
        scheduler,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float = 7.5,
        null_embeddings: Optional[torch.Tensor] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Perform one compensation sampling step.
        
        Args:
            model: UNet model
            scheduler: DDIM scheduler
            latents: Current latents x_t
            timestep: Current timestep t
            text_embeddings: Text condition embeddings
            guidance_scale: Guidance scale w
            null_embeddings: Null condition embeddings
            eta: DDIM eta parameter
            
        Returns:
            Next latents x_{t-1} with compensation
        """
        # Step 1: Compute original conditional noise prediction
        noise_pred_cond, noise_pred_uncond = self.compute_noise_prediction(
            model, latents, timestep, text_embeddings, guidance_scale, null_embeddings
        )
        
        # Step 2: Generate temporary latent estimate x̄_{t-1}
        # Get scheduler coefficients
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        
        # Handle previous timestep
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        
        # DDIM coefficients
        gamma_t = torch.sqrt(alpha_prod_t_prev / alpha_prod_t)
        phi_t = torch.sqrt(alpha_prod_t_prev) - gamma_t * torch.sqrt(alpha_prod_t)
        
        # Temporary latent
        temp_latents = gamma_t * latents + phi_t * noise_pred_cond
        
        # Step 3: Compute null condition compensation term
        if null_embeddings is None:
            null_embeddings = torch.zeros_like(text_embeddings)
            
        noise_pred_null_comp = model(temp_latents, timestep, encoder_hidden_states=null_embeddings).sample
        
        # Step 4: Weighted fusion
        final_noise = self.p * noise_pred_cond + (1 - self.p) * noise_pred_null_comp
        
        # Step 5: Update latents
        next_latents = gamma_t * latents + phi_t * final_noise
        
        return next_latents


def create_compensation_sampler(p: float = 0.8) -> CompensationSampler:
    """
    Factory function to create a CompensationSampler instance.
    
    Args:
        p (float): Compensation parameter
        
    Returns:
        CompensationSampler instance
    """
    return CompensationSampler(p=p)
