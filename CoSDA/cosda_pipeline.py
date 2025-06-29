"""
CoSDA Enhanced Stable Diffusion Pipeline

This module implements the complete CoSDA pipeline that integrates:
1. Compensation Sampling (CoS) for improved forward sampling
2. Drift Alignment Network (DA) for correcting latent drift
3. Enhanced DDIM inversion with better accuracy

The pipeline is designed to work with Tree-Ring watermarking and other
inversion-based watermarking methods.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
import numpy as np
from PIL import Image
import copy

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging, BaseOutput
from diffusers.schedulers import DDIMScheduler

from .compensation_sampling import CompensationSampler
from .drift_alignment import DriftAlignmentNetwork
from .schedulers import CoSDADDIMScheduler

logger = logging.get_logger(__name__)


class CoSDAStableDiffusionPipelineOutput(BaseOutput):
    """Output class for CoSDA Stable Diffusion Pipeline."""
    images: Union[List[Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]
    compensation_metrics: Optional[Dict[str, float]]


class CoSDAStableDiffusionPipeline(StableDiffusionPipeline):
    """
    CoSDA Enhanced Stable Diffusion Pipeline.
    
    Integrates compensation sampling and drift alignment for improved
    inversion-based watermarking robustness.
    """
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        compensation_p: float = 0.8,
        drift_alignment_network: Optional[DriftAlignmentNetwork] = None,
    ):
        """
        Initialize CoSDA Pipeline.
        
        Args:
            compensation_p (float): Compensation parameter for CoS sampling
            drift_alignment_network (DriftAlignmentNetwork): Pre-trained DA network
            Other args: Same as StableDiffusionPipeline
        """
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        
        # Initialize CoSDA components
        self.compensation_sampler = CompensationSampler(p=compensation_p)
        self.drift_alignment_network = drift_alignment_network
        
        # Replace scheduler with CoSDA version if needed
        if not isinstance(scheduler, CoSDADDIMScheduler):
            self.scheduler = CoSDADDIMScheduler.from_config(scheduler.config)
            self.scheduler.compensation_p = compensation_p
        
        self._enable_compensation = False
        self._enable_drift_alignment = drift_alignment_network is not None
        
    def enable_compensation_sampling(self, p: Optional[float] = None):
        """Enable compensation sampling mode."""
        self._enable_compensation = True
        if p is not None:
            self.compensation_sampler.p = p
            if hasattr(self.scheduler, 'enable_compensation_sampling'):
                self.scheduler.enable_compensation_sampling(p)
                
    def disable_compensation_sampling(self):
        """Disable compensation sampling mode."""
        self._enable_compensation = False
        if hasattr(self.scheduler, 'disable_compensation_sampling'):
            self.scheduler.disable_compensation_sampling()
    
    def set_drift_alignment_network(self, network: DriftAlignmentNetwork):
        """Set or update the drift alignment network."""
        self.drift_alignment_network = network
        self._enable_drift_alignment = True
        
    def get_null_text_embeddings(self, batch_size: int = 1) -> torch.Tensor:
        """Get null (empty prompt) text embeddings."""
        null_prompt = [""] * batch_size
        null_input = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        null_embeddings = self.text_encoder(null_input.input_ids.to(self.device))[0]
        return null_embeddings
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, any]] = None,
        enable_compensation: bool = False,
        compensation_p: Optional[float] = None,
    ):
        """
        Enhanced generation with optional compensation sampling.
        
        Args:
            enable_compensation (bool): Whether to use compensation sampling
            compensation_p (float): Compensation parameter override
            Other args: Same as StableDiffusionPipeline.__call__
        """
        # Set compensation mode
        if enable_compensation:
            self.enable_compensation_sampling(compensation_p)
        
        # Standard pipeline setup
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        
        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # Store initial latents
        init_latents = copy.deepcopy(latents)
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Get null embeddings for compensation
        null_embeddings = None
        if self._enable_compensation:
            null_embeddings = self.get_null_text_embeddings(batch_size * num_images_per_prompt)
        
        # Denoising loop
        compensation_metrics = {"total_compensation_steps": 0, "avg_compensation_strength": 0.0}
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Apply compensation sampling if enabled
                compensation_output = None
                if self._enable_compensation and null_embeddings is not None:
                    # Compute temporary latent for compensation
                    temp_latents = self.compensation_sampler.compensation_step(
                        model=self.unet,
                        scheduler=self.scheduler,
                        latents=latents,
                        timestep=t,
                        text_embeddings=prompt_embeds,
                        guidance_scale=guidance_scale,
                        null_embeddings=null_embeddings,
                        eta=eta
                    )
                    
                    # Get null condition prediction on temporary latent
                    compensation_output = self.unet(
                        temp_latents,
                        t,
                        encoder_hidden_states=null_embeddings,
                    ).sample
                    
                    compensation_metrics["total_compensation_steps"] += 1
                
                # Compute previous noisy sample
                if hasattr(self.scheduler, 'step') and 'compensation_output' in self.scheduler.step.__code__.co_varnames:
                    # Use enhanced scheduler with compensation
                    latents = self.scheduler.step(
                        noise_pred, t, latents, 
                        compensation_output=compensation_output,
                        **extra_step_kwargs
                    ).prev_sample
                else:
                    # Use standard scheduler
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Call callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # Calculate compensation metrics
        if compensation_metrics["total_compensation_steps"] > 0:
            compensation_metrics["avg_compensation_strength"] = self.compensation_sampler.p
        
        # Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
            
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # Reset compensation mode
        if enable_compensation:
            self.disable_compensation_sampling()
        
        if not return_dict:
            return (image, has_nsfw_concept)
            
        return CoSDAStableDiffusionPipelineOutput(
            images=image,
            nsfw_content_detected=has_nsfw_concept,
            init_latents=init_latents,
            compensation_metrics=compensation_metrics,
        )

    @torch.no_grad()
    def ddim_inversion(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        apply_drift_alignment: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform DDIM inversion with optional drift alignment.

        Args:
            image: Input image tensor
            num_inference_steps: Number of inversion steps
            guidance_scale: Guidance scale (typically 1.0 for inversion)
            eta: DDIM eta parameter
            generator: Random generator
            return_dict: Whether to return dict
            apply_drift_alignment: Whether to apply drift alignment correction

        Returns:
            Inverted latents or dict with metrics
        """
        device = self._execution_device

        # Encode image to latents
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device=device, dtype=self.vae.dtype)

        latents = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor

        # Prepare for inversion
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Get null embeddings for inversion
        null_embeddings = self.get_null_text_embeddings(latents.shape[0])

        # Inversion loop
        inversion_metrics = {"inversion_error": 0.0, "drift_correction_applied": False}

        for i, t in enumerate(timesteps):
            # Predict noise
            noise_pred = self.unet(latents, t, encoder_hidden_states=null_embeddings).sample

            # Inversion step
            if hasattr(self.scheduler, 'invert_step'):
                latents = self.scheduler.invert_step(noise_pred, t, latents, eta=eta).prev_sample
            else:
                # Manual inversion step
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                next_timestep = t + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_timestep < len(self.scheduler.alphas_cumprod) else torch.tensor(0.0)

                # Predict x_0
                pred_original_sample = (latents - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                # Compute next sample
                pred_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
                latents = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction

        # Apply drift alignment if enabled and network is available
        if apply_drift_alignment and self._enable_drift_alignment and self.drift_alignment_network is not None:
            original_latents = latents.clone()
            latents = self.drift_alignment_network(latents)

            # Calculate correction strength
            correction_strength = torch.norm(latents - original_latents).item()
            inversion_metrics["drift_correction_applied"] = True
            inversion_metrics["correction_strength"] = correction_strength

        if not return_dict:
            return latents

        return {
            "latents": latents,
            "metrics": inversion_metrics
        }
