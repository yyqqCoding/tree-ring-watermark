#!/usr/bin/env python3
"""
模型缓存检查和管理工具

这个脚本帮助你：
1. 查找 Hugging Face 模型的缓存位置
2. 显示已下载的模型文件
3. 提供上传到服务器的指导
4. 管理模型缓存

使用方法:
python check_model_cache.py --model_id runwayml/stable-diffusion-v1-5
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import json
from typing import List, Dict, Optional

def get_hf_cache_dir() -> str:
    """获取 Hugging Face 缓存目录"""
    # 检查环境变量
    if 'HF_HOME' in os.environ:
        return os.path.join(os.environ['HF_HOME'], 'hub')
    elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
        return os.environ['HUGGINGFACE_HUB_CACHE']
    else:
        # 默认位置
        if os.name == 'nt':  # Windows
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
        else:  # Linux/Mac
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
        return cache_dir

def find_model_cache(model_id: str) -> Optional[str]:
    """查找特定模型的缓存目录"""
    cache_dir = get_hf_cache_dir()
    
    # 将模型ID转换为缓存目录名
    # 例如: runwayml/stable-diffusion-v1-5 -> models--runwayml--stable-diffusion-v1-5
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, cache_name)
    
    if os.path.exists(model_cache_path):
        return model_cache_path
    else:
        return None

def get_model_files(model_cache_path: str) -> Dict[str, List[str]]:
    """获取模型缓存目录中的文件列表"""
    files_info = {
        'snapshots': [],
        'refs': [],
        'blobs': []
    }
    
    if not os.path.exists(model_cache_path):
        return files_info
    
    # 检查 snapshots 目录
    snapshots_dir = os.path.join(model_cache_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        for snapshot in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_path):
                files_info['snapshots'].append(snapshot)
    
    # 检查 refs 目录
    refs_dir = os.path.join(model_cache_path, 'refs')
    if os.path.exists(refs_dir):
        files_info['refs'] = os.listdir(refs_dir)
    
    # 检查 blobs 目录
    blobs_dir = os.path.join(model_cache_path, 'blobs')
    if os.path.exists(blobs_dir):
        files_info['blobs'] = os.listdir(blobs_dir)
    
    return files_info

def calculate_cache_size(model_cache_path: str) -> int:
    """计算模型缓存的总大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_cache_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"

def print_model_info(model_id: str):
    """打印模型缓存信息"""
    print(f"\n{'='*60}")
    print(f"模型: {model_id}")
    print(f"{'='*60}")
    
    # 查找缓存目录
    cache_path = find_model_cache(model_id)
    if not cache_path:
        print(f"❌ 未找到模型 {model_id} 的缓存")
        print(f"💡 请先运行脚本下载模型，或检查模型ID是否正确")
        return
    
    print(f"✅ 找到模型缓存")
    print(f"📁 缓存路径: {cache_path}")
    
    # 计算大小
    cache_size = calculate_cache_size(cache_path)
    print(f"💾 缓存大小: {format_size(cache_size)}")
    
    # 获取文件信息
    files_info = get_model_files(cache_path)
    
    print(f"\n📋 缓存内容:")
    print(f"  - Snapshots: {len(files_info['snapshots'])} 个")
    print(f"  - Refs: {len(files_info['refs'])} 个")
    print(f"  - Blobs: {len(files_info['blobs'])} 个")
    
    # 显示最新的 snapshot
    if files_info['snapshots']:
        latest_snapshot = max(files_info['snapshots'])
        snapshot_path = os.path.join(cache_path, 'snapshots', latest_snapshot)
        print(f"\n📂 最新 Snapshot: {latest_snapshot}")
        print(f"   路径: {snapshot_path}")
        
        # 列出 snapshot 中的文件
        if os.path.exists(snapshot_path):
            snapshot_files = []
            for root, dirs, files in os.walk(snapshot_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), snapshot_path)
                    file_size = os.path.getsize(os.path.join(root, file))
                    snapshot_files.append((rel_path, file_size))
            
            print(f"   包含 {len(snapshot_files)} 个文件:")
            for file_path, file_size in sorted(snapshot_files):
                print(f"     - {file_path} ({format_size(file_size)})")

def print_upload_instructions(model_id: str):
    """打印上传到服务器的说明"""
    cache_path = find_model_cache(model_id)
    if not cache_path:
        print("❌ 未找到模型缓存，无法提供上传说明")
        return
    
    print(f"\n{'='*60}")
    print("📤 上传到服务器的说明")
    print(f"{'='*60}")
    
    print("1. 压缩模型缓存:")
    cache_name = os.path.basename(cache_path)
    print(f"   cd {os.path.dirname(cache_path)}")
    print(f"   tar -czf {cache_name}.tar.gz {cache_name}")
    
    print("\n2. 上传到服务器:")
    print(f"   scp {cache_name}.tar.gz user@server:/path/to/server/")
    
    print("\n3. 在服务器上解压:")
    server_cache_dir = "~/.cache/huggingface/hub"
    print(f"   cd {server_cache_dir}")
    print(f"   tar -xzf {cache_name}.tar.gz")
    
    print("\n4. 验证:")
    print(f"   ls -la {server_cache_dir}/{cache_name}")
    
    print(f"\n💡 提示:")
    print(f"   - 服务器上的缓存目录通常是: ~/.cache/huggingface/hub")
    print(f"   - 如果服务器使用不同的缓存目录，请设置环境变量:")
    print(f"     export HF_HOME=/your/custom/cache/dir")
    print(f"   - 确保服务器上有足够的磁盘空间 ({format_size(calculate_cache_size(cache_path))})")

def main():
    parser = argparse.ArgumentParser(description="检查和管理 Hugging Face 模型缓存")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", 
                       help="模型ID (默认: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--list_all", action="store_true", 
                       help="列出所有缓存的模型")
    parser.add_argument("--cache_dir", action="store_true", 
                       help="显示缓存目录位置")
    
    args = parser.parse_args()
    
    if args.cache_dir:
        cache_dir = get_hf_cache_dir()
        print(f"🗂️  Hugging Face 缓存目录: {cache_dir}")
        print(f"📊 目录是否存在: {'✅' if os.path.exists(cache_dir) else '❌'}")
        return
    
    if args.list_all:
        cache_dir = get_hf_cache_dir()
        if not os.path.exists(cache_dir):
            print(f"❌ 缓存目录不存在: {cache_dir}")
            return
        
        print(f"📋 所有缓存的模型:")
        models = [d for d in os.listdir(cache_dir) if d.startswith('models--')]
        if not models:
            print("   (无缓存模型)")
        else:
            for model in sorted(models):
                model_id = model.replace('models--', '').replace('--', '/')
                model_path = os.path.join(cache_dir, model)
                size = calculate_cache_size(model_path)
                print(f"   - {model_id} ({format_size(size)})")
        return
    
    # 检查特定模型
    print_model_info(args.model_id)
    print_upload_instructions(args.model_id)

if __name__ == "__main__":
    main()
