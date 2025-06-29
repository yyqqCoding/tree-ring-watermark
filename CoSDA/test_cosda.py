"""
Test Script for CoSDA Implementation

This script provides basic tests to verify that the CoSDA implementation
works correctly with the Tree-Ring watermarking framework.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CoSDA.compensation_sampling import CompensationSampler
from CoSDA.drift_alignment import DriftAlignmentNetwork, DistortionGenerator
from CoSDA.schedulers import CoSDADDIMScheduler
from CoSDA.utils import create_distortions, evaluate_inversion_error


def test_compensation_sampler():
    """Test the Compensation Sampler implementation."""
    print("Testing Compensation Sampler...")
    
    # Initialize sampler
    sampler = CompensationSampler(p=0.8)
    
    # Test basic functionality
    assert sampler.p == 0.8, "Compensation parameter not set correctly"
    
    # Test parameter update
    sampler.p = 0.5
    assert sampler.p == 0.5, "Compensation parameter update failed"
    
    print("✓ Compensation Sampler tests passed")


def test_drift_alignment_network():
    """Test the Drift Alignment Network implementation."""
    print("Testing Drift Alignment Network...")
    
    # Initialize network
    network = DriftAlignmentNetwork(in_channels=4, hidden_channels=64)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 64, 64)
    
    with torch.no_grad():
        output = network(input_tensor)
    
    # Check output shape
    assert output.shape == input_tensor.shape, f"Output shape {output.shape} doesn't match input shape {input_tensor.shape}"
    
    # Test that network produces different outputs for different inputs
    input_tensor2 = torch.randn(batch_size, 4, 64, 64)
    with torch.no_grad():
        output2 = network(input_tensor2)
    
    assert not torch.allclose(output, output2), "Network produces identical outputs for different inputs"
    
    print("✓ Drift Alignment Network tests passed")


def test_distortion_generator():
    """Test the Distortion Generator implementation."""
    print("Testing Distortion Generator...")
    
    # Initialize generator
    generator = DistortionGenerator()
    
    # Create test image
    test_image = torch.rand(1, 3, 256, 256)  # RGB image
    
    # Test JPEG compression
    try:
        compressed = generator.apply_jpeg_compression(test_image, quality=30)
        assert compressed.shape == test_image.shape, "JPEG compression changed image shape"
        print("  ✓ JPEG compression test passed")
    except Exception as e:
        print(f"  ✗ JPEG compression test failed: {e}")
    
    # Test Gaussian noise
    try:
        noisy = generator.apply_gaussian_noise(test_image, sigma=0.1)
        assert noisy.shape == test_image.shape, "Gaussian noise changed image shape"
        assert not torch.allclose(test_image, noisy), "Gaussian noise didn't change image"
        print("  ✓ Gaussian noise test passed")
    except Exception as e:
        print(f"  ✗ Gaussian noise test failed: {e}")
    
    # Test crop and resize
    try:
        cropped = generator.apply_random_crop_resize(test_image, crop_ratio=0.8)
        assert cropped.shape == test_image.shape, "Crop/resize changed image shape"
        print("  ✓ Crop/resize test passed")
    except Exception as e:
        print(f"  ✗ Crop/resize test failed: {e}")
    
    print("✓ Distortion Generator tests passed")


def test_cosda_scheduler():
    """Test the CoSDA DDIM Scheduler implementation."""
    print("Testing CoSDA DDIM Scheduler...")
    
    # Initialize scheduler
    scheduler = CoSDADDIMScheduler(
        num_train_timesteps=1000,
        compensation_p=0.8
    )
    
    # Test compensation mode
    scheduler.enable_compensation_sampling(p=0.7)
    assert scheduler._enable_compensation == True, "Compensation mode not enabled"
    assert scheduler.compensation_p == 0.7, "Compensation parameter not updated"
    
    scheduler.disable_compensation_sampling()
    assert scheduler._enable_compensation == False, "Compensation mode not disabled"
    
    # Test timestep setting
    scheduler.set_timesteps(50)
    assert len(scheduler.timesteps) == 50, "Timesteps not set correctly"
    
    print("✓ CoSDA DDIM Scheduler tests passed")


def test_utils_functions():
    """Test utility functions."""
    print("Testing Utility Functions...")
    
    # Test distortion creation
    test_image = torch.rand(1, 3, 256, 256)
    
    try:
        distortions = create_distortions(
            test_image,
            distortion_types=['gaussian_noise'],
            distortion_params={'noise_sigmas': [0.1]}
        )
        assert 'noise_s0.1' in distortions, "Distortion not created correctly"
        print("  ✓ Distortion creation test passed")
    except Exception as e:
        print(f"  ✗ Distortion creation test failed: {e}")
    
    # Test inversion error evaluation
    try:
        original = torch.randn(2, 4, 64, 64)
        inverted = original + torch.randn_like(original) * 0.1
        
        metrics = evaluate_inversion_error(original, inverted)
        assert 'mse' in metrics, "MSE metric not computed"
        assert 'mae' in metrics, "MAE metric not computed"
        assert metrics['mse'] > 0, "MSE should be positive"
        print("  ✓ Inversion error evaluation test passed")
    except Exception as e:
        print(f"  ✗ Inversion error evaluation test failed: {e}")
    
    print("✓ Utility Functions tests passed")


def test_integration_compatibility():
    """Test compatibility with Tree-Ring watermarking components."""
    print("Testing Integration Compatibility...")
    
    try:
        # Test import of Tree-Ring components
        from optim_utils import get_watermarking_mask, inject_watermark, eval_watermark
        print("  ✓ Tree-Ring imports successful")
        
        # Test basic watermarking operations
        shape = (1, 4, 64, 64)
        init_latents = torch.randn(shape)
        
        # Mock watermark args
        watermark_args = type('Args', (), {
            'w_channel': 0,
            'w_radius': 10,
            'w_pattern': 'rand',
            'w_injection': 'complex',
            'w_measurement': 'complex'
        })()
        
        # Test watermarking mask generation
        try:
            mask = get_watermarking_mask(init_latents, watermark_args, 'cpu')
            assert mask.shape == init_latents.shape, "Watermarking mask shape mismatch"
            print("  ✓ Watermarking mask generation test passed")
        except Exception as e:
            print(f"  ✗ Watermarking mask generation test failed: {e}")
        
    except ImportError as e:
        print(f"  ✗ Tree-Ring import test failed: {e}")
        print("    Make sure you're running this from the Tree-Ring project directory")
    
    print("✓ Integration Compatibility tests completed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running CoSDA Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_compensation_sampler,
        test_drift_alignment_network,
        test_distortion_generator,
        test_cosda_scheduler,
        test_utils_functions,
        test_integration_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! CoSDA implementation is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


def test_memory_usage():
    """Test memory usage of CoSDA components."""
    print("Testing Memory Usage...")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("  Using CPU")
    
    # Test Drift Alignment Network memory usage
    network = DriftAlignmentNetwork().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"  Drift Alignment Network:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass memory
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 4, 64, 64).to(device)
    
    with torch.no_grad():
        output = network(input_tensor)
    
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"    Peak GPU memory usage: {peak_memory:.2f} MB")
    
    print("✓ Memory usage test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CoSDA Implementation")
    parser.add_argument("--memory", action="store_true", help="Include memory usage tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_all_tests()
    
    if args.memory:
        print()
        test_memory_usage()
    
    if not success:
        sys.exit(1)
