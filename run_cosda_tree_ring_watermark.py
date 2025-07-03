"""
CoSDA Tree-Ring Watermarking Evaluation Script

This script runs the same evaluation as run_tree_ring_watermark.py but with CoSDA optimizations.
It provides a direct comparison between baseline Tree-Ring and CoSDA-enhanced methods.

Key Features:
1. CoSDA-enhanced Stable Diffusion pipeline for improved robustness
2. Compensation Sampling (CoS) to reduce condition mismatch errors
3. Drift Alignment Network (DA) to correct latent feature drift
4. Comprehensive evaluation with wandb tracking
5. Support for various watermark patterns and distortions

Author: Based on CoSDA paper implementation
Date: 2024
"""

import argparse
import torch
import numpy as np
import wandb
from tqdm import tqdm
import copy
import os
import sys
import logging

# 设置详细的日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('cosda_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cosda_evaluation.log'),
        logging.StreamHandler()
    ]
)

# Import existing Tree-Ring components
from optim_utils import *
from io_utils import *

# Import CoSDA components
from CoSDA import CoSDAStableDiffusionPipeline, DriftAlignmentNetwork
from CoSDA.tree_ring_integration import CoSDATreeRingWatermarker
from CoSDA.utils import create_cosda_config
from diffusers import DDIMScheduler
import open_clip


def get_reference_model(reference_model, reference_model_pretrain, device):
    """Load reference model for CLIP evaluation."""
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        reference_model,
        pretrained=reference_model_pretrain,
        device=device
    )
    ref_tokenizer = open_clip.get_tokenizer(reference_model)
    return ref_model, ref_clip_preprocess, ref_tokenizer


def load_cosda_pipeline(args):
    """
    Load CoSDA-enhanced Stable Diffusion pipeline.

    Args:
        args: Command line arguments containing model configuration

    Returns:
        tuple: (pipeline, drift_alignment_network)
    """
    logger.info("开始加载 CoSDA 增强的 Stable Diffusion 管道...")
    logger.info(f"模型ID: {args.model_id}")
    logger.info(f"设备: {args.device}")
    logger.info(f"数据类型: {'float16' if args.device == 'cuda' else 'float32'}")

    # 加载调度器
    logger.info("正在加载 DDIM 调度器...")
    try:
        scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler", local_files_only=True)
        logger.info("从本地缓存加载 DDIM 调度器成功")
    except Exception as e:
        logger.warning(f"从本地缓存加载调度器失败: {e}")
        logger.info("尝试使用默认调度器配置...")
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )
    logger.info("DDIM 调度器加载完成")

    # 加载 CoSDA 管道
    logger.info("正在加载 CoSDA 管道组件...")
    try:
        pipeline = CoSDAStableDiffusionPipeline.from_pretrained(
            args.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,  # 只使用本地文件
            low_cpu_mem_usage=True,  # 减少CPU内存使用
        )
        logger.info("从本地缓存加载管道成功")
    except Exception as e:
        logger.error(f"从本地缓存加载管道失败: {e}")
        raise RuntimeError("无法加载模型，请检查本地缓存或网络连接")

    # 启用内存优化（检查方法是否存在）
    if args.device == "cuda":
        try:
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()  # 将模型组件卸载到CPU
                logger.info("已启用模型CPU卸载")
        except Exception as e:
            logger.warning(f"无法启用模型CPU卸载: {e}")

        try:
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()  # 启用注意力切片
                logger.info("已启用注意力切片")
        except Exception as e:
            logger.warning(f"无法启用注意力切片: {e}")

        try:
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()  # 启用VAE切片
                logger.info("已启用VAE切片")
        except Exception as e:
            logger.warning(f"无法启用VAE切片: {e}")

        logger.info("内存优化设置完成")

    pipeline = pipeline.to(args.device)
    logger.info(f"CoSDA 管道已加载到设备: {args.device}")
    
    # 加载漂移对齐网络（如果可用）
    drift_alignment_network = None
    if args.drift_alignment_checkpoint and os.path.exists(args.drift_alignment_checkpoint):
        logger.info(f"正在从 {args.drift_alignment_checkpoint} 加载漂移对齐网络...")
        drift_alignment_network = DriftAlignmentNetwork(in_channels=4, hidden_channels=64)
        checkpoint = torch.load(args.drift_alignment_checkpoint, map_location=args.device)
        drift_alignment_network.load_state_dict(checkpoint['model_state_dict'])
        drift_alignment_network = drift_alignment_network.to(args.device)
        drift_alignment_network.eval()
        logger.info("漂移对齐网络加载成功")
    else:
        logger.warning("未找到漂移对齐网络检查点，使用基础 CoSDA")
        logger.info(f"检查点路径: {args.drift_alignment_checkpoint}")

    logger.info("CoSDA 管道加载完成")
    return pipeline, drift_alignment_network


def main(args):
    """
    主要的评估函数，使用 CoSDA 优化。

    Args:
        args: 命令行参数，包含所有配置选项
    """
    logger.info("="*60)
    logger.info("开始 CoSDA Tree-Ring 水印评估")
    logger.info("="*60)
    logger.info(f"运行名称: {args.run_name}")
    logger.info(f"评估范围: {args.start} - {args.end} (共 {args.end - args.start} 个样本)")
    logger.info(f"水印通道: {args.w_channel}")
    logger.info(f"水印模式: {args.w_pattern}")
    logger.info(f"水印半径: {args.w_radius}")
    logger.info(f"补偿参数: {args.compensation_p}")
    logger.info(f"推理步数: {args.num_inference_steps}")
    logger.info(f"引导尺度: {args.guidance_scale}")
    logger.info(f"设备: {args.device}")

    # 设置随机种子
    logger.info(f"设置随机种子: {args.gen_seed}")
    set_random_seed(args.gen_seed)

    # 加载 CoSDA 管道
    logger.info("正在加载 CoSDA 管道...")
    pipeline, drift_alignment_network = load_cosda_pipeline(args)
    
    # Initialize CoSDA watermarker
    watermarker = CoSDATreeRingWatermarker(
        pipeline=pipeline,
        drift_alignment_network=drift_alignment_network,
        compensation_p=args.compensation_p,
        device=args.device
    )
    
    # Load reference model for CLIP evaluation
    if args.reference_model is not None:
        ref_model, ref_clip_preprocess, ref_tokenizer = get_reference_model(args.reference_model, args.reference_model_pretrain, args.device)
    
    # Initialize tracking
    if args.with_tracking:
        wandb.init(
            project="cosda-tree-ring-watermark",
            name=f"cosda_{args.run_name}",
            config=vars(args)
        )
        table = wandb.Table(columns=["no_w_image", "w_image", "no_w_clip_score", "w_clip_score", "prompt"])
    
    # Load prompts
    if args.dataset == 'Gustavosta/Stable-Diffusion-Prompts':
        dataset = load_dataset(args.dataset)['train']['Prompt']
        prompts = [dataset[i] for i in range(args.start, min(args.end, len(dataset)))]
    else:
        prompts = [f"prompt_{i}" for i in range(args.start, args.end)]
    
    # Watermark configuration
    watermark_args = {
        'w_seed': args.w_seed,
        'w_channel': args.w_channel,
        'w_radius': args.w_radius,
        'w_pattern': args.w_pattern,
        'w_mask_shape': args.w_mask_shape,
        'w_injection': args.w_injection,
        'w_measurement': args.w_measurement,
        'w_pattern_const': args.w_pattern_const
    }
    
    # Results storage
    results = []
    no_w_clip_scores = []
    w_clip_scores = []
    
    print(f"Starting CoSDA evaluation for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(tqdm(prompts)):
        try:
            # Set seed for reproducibility
            generator = torch.Generator(device=args.device).manual_seed(args.gen_seed + i)
            
            ### Generation Phase ###
            
            # Generate without watermark (baseline)
            watermarker.pipeline.disable_compensation_sampling()
            no_w_result = watermarker.generate_watermarked_image(
                prompt=prompt,
                watermark_args=watermark_args,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                enable_compensation=False
            )
            orig_image_no_w = no_w_result['image']
            
            # Generate with watermark using CoSDA
            generator = torch.Generator(device=args.device).manual_seed(args.gen_seed + i)  # Reset seed
            watermarker.pipeline.enable_compensation_sampling(args.compensation_p)
            w_result = watermarker.generate_watermarked_image(
                prompt=prompt,
                watermark_args=watermark_args,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                enable_compensation=True
            )
            orig_image_w = w_result['image']
            
            ### Evaluation Phase ###
            
            # Apply distortions if specified
            if args.attack_mode != 'no_attack':
                orig_image_no_w_auged, orig_image_w_auged = image_distortion(
                    orig_image_no_w, orig_image_w, args.gen_seed + i, args
                )
            else:
                orig_image_no_w_auged = orig_image_no_w
                orig_image_w_auged = orig_image_w
            
            # Convert to tensors for watermark extraction
            img_no_w_tensor = transform_img(orig_image_no_w_auged).unsqueeze(0).to(args.device)
            img_w_tensor = transform_img(orig_image_w_auged).unsqueeze(0).to(args.device)
            
            # Extract watermarks using CoSDA
            no_w_extraction = watermarker.extract_watermark(
                img_no_w_tensor,
                no_w_result['watermarking_mask'],
                no_w_result['gt_patch'],
                watermark_args,
                num_inversion_steps=args.test_num_inference_steps,
                apply_drift_alignment=True
            )
            
            w_extraction = watermarker.extract_watermark(
                img_w_tensor,
                w_result['watermarking_mask'],
                w_result['gt_patch'],
                watermark_args,
                num_inversion_steps=args.test_num_inference_steps,
                apply_drift_alignment=True
            )
            
            no_w_metric = -no_w_extraction['watermark_score']  # Negative for consistency
            w_metric = -w_extraction['watermark_score']
            
            # Compute CLIP scores if reference model provided
            if args.reference_model is not None:
                sims = measure_similarity(
                    [orig_image_no_w_auged, orig_image_w_auged], 
                    prompt, 
                    ref_model, 
                    ref_clip_preprocess, 
                    ref_tokenizer, 
                    args.device
                )
                no_w_clip_score = sims[0].item()
                w_clip_score = sims[1].item()
            else:
                no_w_clip_score = 0
                w_clip_score = 0
            
            # Store results
            results.append({
                'no_w_metric': no_w_metric,
                'w_metric': w_metric,
                'no_w_clip_score': no_w_clip_score,
                'w_clip_score': w_clip_score,
                'compensation_metrics': w_result.get('compensation_metrics', {}),
                'inversion_metrics': w_extraction.get('inversion_metrics', {})
            })
            
            no_w_clip_scores.append(no_w_clip_score)
            w_clip_scores.append(w_clip_score)
            
            # Log to wandb
            if args.with_tracking:
                if i < args.max_num_log_image:
                    table.add_data(
                        wandb.Image(orig_image_no_w_auged),
                        wandb.Image(orig_image_w_auged),
                        no_w_clip_score,
                        w_clip_score,
                        prompt
                    )
                else:
                    table.add_data(None, None, no_w_clip_score, w_clip_score, prompt)
                
                # Log metrics
                wandb.log({
                    'step': i,
                    'no_w_metric': no_w_metric,
                    'w_metric': w_metric,
                    'no_w_clip_score': no_w_clip_score,
                    'w_clip_score': w_clip_score,
                    'compensation_strength': w_result.get('compensation_metrics', {}).get('avg_compensation_strength', 0),
                    'drift_correction_applied': w_extraction.get('inversion_metrics', {}).get('drift_correction_applied', False)
                })
        
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            continue
    
    # Compute final statistics
    final_results = {
        'no_w_metrics': [r['no_w_metric'] for r in results],
        'w_metrics': [r['w_metric'] for r in results],
        'no_w_clip_scores': no_w_clip_scores,
        'w_clip_scores': w_clip_scores
    }
    
    # Print summary
    print("\n" + "="*60)
    print("CoSDA Tree-Ring Watermarking Results")
    print("="*60)
    print(f"Number of samples: {len(results)}")
    print(f"Watermark detection (mean ± std): {np.mean(final_results['w_metrics']):.4f} ± {np.std(final_results['w_metrics']):.4f}")
    print(f"No watermark detection (mean ± std): {np.mean(final_results['no_w_metrics']):.4f} ± {np.std(final_results['no_w_metrics']):.4f}")
    
    if args.reference_model is not None:
        print(f"CLIP Score - No watermark (mean ± std): {np.mean(no_w_clip_scores):.4f} ± {np.std(no_w_clip_scores):.4f}")
        print(f"CLIP Score - With watermark (mean ± std): {np.mean(w_clip_scores):.4f} ± {np.std(w_clip_scores):.4f}")
        print(f"CLIP Score difference: {np.mean(w_clip_scores) - np.mean(no_w_clip_scores):.4f}")
    
    # Log final results
    if args.with_tracking:
        wandb.log({
            'final/w_metric_mean': np.mean(final_results['w_metrics']),
            'final/w_metric_std': np.std(final_results['w_metrics']),
            'final/no_w_metric_mean': np.mean(final_results['no_w_metrics']),
            'final/no_w_metric_std': np.std(final_results['no_w_metrics']),
            'final/clip_score_w_mean': np.mean(w_clip_scores),
            'final/clip_score_w_std': np.std(w_clip_scores),
            'final/clip_score_no_w_mean': np.mean(no_w_clip_scores),
            'final/clip_score_no_w_std': np.std(no_w_clip_scores),
            'final/clip_score_diff': np.mean(w_clip_scores) - np.mean(no_w_clip_scores)
        })
        
        wandb.log({"results_table": table})
        wandb.finish()
    
    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoSDA Tree-Ring Watermarking Evaluation')
    
    # Model and device settings
    parser.add_argument('--model_id', default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    
    # CoSDA specific settings
    parser.add_argument('--compensation_p', type=float, default=0.8, help='Compensation parameter for CoS sampling')
    parser.add_argument('--drift_alignment_checkpoint', default='./cosda_checkpoints/best_drift_alignment.pth', help='Path to drift alignment checkpoint')
    
    # Watermarking settings
    parser.add_argument('--w_seed', type=int, default=999999)
    parser.add_argument('--w_channel', type=int, default=0)
    parser.add_argument('--w_radius', type=int, default=10)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', type=float, default=0)
    
    # Generation settings
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--test_num_inference_steps', type=int, default=50)
    
    # Evaluation settings
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--gen_seed', type=int, default=0)
    parser.add_argument('--run_name', default='cosda_evaluation')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--attack_mode', default='no_attack')
    parser.add_argument('--image_length', type=int, default=512)
    parser.add_argument('--num_images', type=int, default=1)

    # Image distortion parameters
    parser.add_argument('--r_degree', type=float, default=None)
    parser.add_argument('--jpeg_ratio', type=int, default=None)
    parser.add_argument('--crop_scale', type=float, default=None)
    parser.add_argument('--crop_ratio', type=float, default=None)
    parser.add_argument('--gaussian_blur_r', type=int, default=None)
    parser.add_argument('--gaussian_std', type=float, default=None)
    parser.add_argument('--brightness_factor', type=float, default=None)
    parser.add_argument('--rand_aug', type=int, default=0)
    
    # Reference model for CLIP evaluation
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    
    # Tracking settings
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--max_num_log_image', type=int, default=100)
    
    args = parser.parse_args()

    # Set default test_num_inference_steps if not provided
    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    main(args)
