"""
CoSDA Enhanced DDIM Scheduler

This module implements an enhanced DDIM scheduler that supports compensation sampling
and improved inversion accuracy for the CoSDA framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
import numpy as np
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput


class CoSDADDIMScheduler(DDIMScheduler):
    """
    Enhanced DDIM Scheduler with Compensation Sampling support.
    
    Extends the standard DDIM scheduler to support:
    1. Compensation sampling mechanism
    2. Improved inversion accuracy
    3. Better handling of condition mismatches
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        compensation_p: float = 0.8,
        **kwargs
    ):
        """
        Initialize CoSDA DDIM Scheduler.
        
        Args:
            compensation_p (float): Compensation parameter for CoS sampling
            Other args: Same as DDIMScheduler
        """
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            **kwargs
        )
        
        self.compensation_p = compensation_p
        self._enable_compensation = False
        
    def enable_compensation_sampling(self, p: Optional[float] = None):
        """Enable compensation sampling mode."""
        self._enable_compensation = True
        if p is not None:
            self.compensation_p = p
            
    def disable_compensation_sampling(self):
        """Disable compensation sampling mode."""
        self._enable_compensation = False
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        compensation_output: Optional[torch.FloatTensor] = None,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Enhanced step function with compensation sampling support.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in diffusion chain
            sample: Current instance of sample being created by diffusion process
            eta: Weight of noise for added noise in diffusion step
            use_clipped_model_output: Whether to clip predicted x_0
            generator: Random number generator
            variance_noise: Alternative to generator for noise
            return_dict: Whether to return dict or tuple
            compensation_output: Additional model output for compensation (null condition)
            
        Returns:
            DDIMSchedulerOutput or tuple with previous sample
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 5. Apply compensation sampling if enabled
        if self._enable_compensation and compensation_output is not None:
            # Weighted combination of original and compensation outputs
            pred_epsilon = self.compensation_p * pred_epsilon + (1 - self.compensation_p) * compensation_output

        # 6. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 8. compute x_t-1 without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    def invert_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Inversion step for DDIM inversion (image to noise).
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in diffusion chain
            sample: Current instance of sample being inverted
            eta: Weight of noise for added noise in inversion step
            use_clipped_model_output: Whether to clip predicted x_0
            generator: Random number generator
            return_dict: Whether to return dict or tuple
            
        Returns:
            DDIMSchedulerOutput or tuple with next sample (more noisy)
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get next step value (=t+1)
        next_timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_next = self.alphas_cumprod[next_timestep] if next_timestep < len(self.alphas_cumprod) else torch.tensor(0.0)

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute "direction pointing to x_t+1" for inversion
        pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * pred_epsilon

        # 6. compute x_t+1 (more noisy sample)
        next_sample = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (next_sample,)

        return DDIMSchedulerOutput(prev_sample=next_sample, pred_original_sample=pred_original_sample)
