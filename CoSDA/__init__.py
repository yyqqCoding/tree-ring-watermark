"""
CoSDA: Enhancing the Robustness of Inversion-based Generative Image Watermarking Framework

This module implements the CoSDA optimization method to improve DDIM inversion accuracy
for Tree-Ring watermarking and other inversion-based watermarking methods.

Key Components:
1. Compensation Sampling (CoS): Reduces condition mismatch errors in forward sampling
2. Drift Alignment Network (DA): Corrects latent feature drift caused by image distortions
3. Enhanced DDIM Inversion: Improved inversion process with better accuracy

Authors: Based on the CoSDA paper implementation
"""

from .compensation_sampling import CompensationSampler
from .drift_alignment import DriftAlignmentNetwork, DriftAlignmentTrainer
from .cosda_pipeline import CoSDAStableDiffusionPipeline
from .utils import create_distortions, evaluate_inversion_error
from .schedulers import CoSDADDIMScheduler

__version__ = "1.0.0"
__all__ = [
    "CompensationSampler",
    "DriftAlignmentNetwork", 
    "DriftAlignmentTrainer",
    "CoSDAStableDiffusionPipeline",
    "CoSDADDIMScheduler",
    "create_distortions",
    "evaluate_inversion_error"
]
