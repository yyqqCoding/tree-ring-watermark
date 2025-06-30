#!/usr/bin/env python3
"""
服务器端模型缓存设置工具

这个脚本帮助在服务器上设置和管理 Hugging Face 模型缓存。

使用方法:
1. 上传模型缓存文件到服务器
2. 运行此脚本解压和设置缓存
3. 验证缓存是否正确设置

python setup_server_cache.py --extract model_cache.tar.gz
"""

import os
import sys
import argparse
import tarfile
import shutil
from pathlib import Path
import json

def get_hf_cache_dir() -> str:
    """获取 Hugging Face 缓存目录"""
    if 'HF_HOME' in os.environ:
        return os.path.join(os.environ['HF_HOME'], 'hub')
    elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
        return os.environ['HUGGINGFACE_HUB_CACHE']
    else:
        return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

def create_cache_dir(cache_dir: str):
    """创建缓存目录"""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ 缓存目录已创建: {cache_dir}")
        return True
    except Exception as e:
        print(f"❌ 创建缓存目录失败: {e}")
        return False

def extract_model_cache(archive_path: str, cache_dir: str):
    """解压模型缓存文件"""
    if not os.path.exists(archive_path):
        print(f"❌ 压缩文件不存在: {archive_path}")
        return False
    
    print(f"📦 正在解压 {archive_path} 到 {cache_dir}...")
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=cache_dir)
        print(f"✅ 解压完成")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def verify_model_cache(model_id: str, cache_dir: str):
    """验证模型缓存是否正确"""
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, cache_name)
    
    print(f"\n🔍 验证模型缓存: {model_id}")
    print(f"📁 预期路径: {model_cache_path}")
    
    if not os.path.exists(model_cache_path):
        print(f"❌ 模型缓存目录不存在")
        return False
    
    # 检查必要的子目录
    required_dirs = ['snapshots', 'refs']
    for dir_name in required_dirs:
        dir_path = os.path.join(model_cache_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name} 目录存在")
        else:
            print(f"⚠️  {dir_name} 目录不存在")
    
    # 检查 snapshots 内容
    snapshots_dir = os.path.join(model_cache_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            print(f"✅ 找到 {len(snapshots)} 个 snapshot")
            latest_snapshot = max(snapshots)
            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
            
            # 检查 snapshot 中的文件
            if os.path.exists(snapshot_path):
                files = []
                for root, dirs, filenames in os.walk(snapshot_path):
                    files.extend(filenames)
                print(f"✅ 最新 snapshot 包含 {len(files)} 个文件")
                
                # 检查关键文件
                key_files = ['model_index.json', 'config.json']
                for key_file in key_files:
                    if any(key_file in f for f in files):
                        print(f"✅ 找到关键文件: {key_file}")
                    else:
                        print(f"⚠️  未找到关键文件: {key_file}")
        else:
            print(f"❌ snapshots 目录为空")
            return False
    
    return True

def set_environment_variables():
    """设置环境变量"""
    cache_dir = get_hf_cache_dir()
    
    print(f"\n🔧 环境变量设置:")
    print(f"当前 HF 缓存目录: {cache_dir}")
    
    # 生成环境变量设置命令
    print(f"\n💡 如需自定义缓存目录，请设置环境变量:")
    print(f"export HF_HOME=/your/custom/cache/dir")
    print(f"export HUGGINGFACE_HUB_CACHE=/your/custom/cache/dir/hub")
    
    # 生成 .bashrc 添加命令
    print(f"\n📝 永久设置 (添加到 ~/.bashrc):")
    print(f"echo 'export HF_HOME={cache_dir.replace('/hub', '')}' >> ~/.bashrc")
    print(f"echo 'export HUGGINGFACE_HUB_CACHE={cache_dir}' >> ~/.bashrc")

def list_available_models(cache_dir: str):
    """列出可用的模型"""
    print(f"\n📋 缓存目录中的模型:")

    if not os.path.exists(cache_dir):
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return

    models = [d for d in os.listdir(cache_dir) if d.startswith('models--')]
    if not models:
        print("   (无缓存模型)")
    else:
        for model in sorted(models):
            model_id = model.replace('models--', '').replace('--', '/')
            model_path = os.path.join(cache_dir, model)

            # 计算大小
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)

            # 格式化大小
            size_str = format_size(total_size)
            print(f"   ✅ {model_id} ({size_str})")

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

def main():
    parser = argparse.ArgumentParser(description="服务器端模型缓存设置工具")
    parser.add_argument("--extract", metavar="ARCHIVE", 
                       help="解压模型缓存压缩文件")
    parser.add_argument("--verify", metavar="MODEL_ID", 
                       help="验证特定模型的缓存")
    parser.add_argument("--list", action="store_true", 
                       help="列出所有缓存的模型")
    parser.add_argument("--setup-env", action="store_true", 
                       help="显示环境变量设置说明")
    parser.add_argument("--cache-dir", 
                       help="自定义缓存目录路径")
    
    args = parser.parse_args()
    
    # 确定缓存目录
    if args.cache_dir:
        cache_dir = args.cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    else:
        cache_dir = get_hf_cache_dir()
    
    print(f"🗂️  使用缓存目录: {cache_dir}")
    
    if args.extract:
        # 创建缓存目录
        if create_cache_dir(cache_dir):
            # 解压模型缓存
            if extract_model_cache(args.extract, cache_dir):
                print(f"✅ 模型缓存设置完成")
                list_available_models(cache_dir)
    
    elif args.verify:
        verify_model_cache(args.verify, cache_dir)
    
    elif args.list:
        list_available_models(cache_dir)
    
    elif args.setup_env:
        set_environment_variables()
    
    else:
        print("请指定操作选项。使用 --help 查看帮助。")
        print("\n常用命令:")
        print("  python setup_server_cache.py --extract model_cache.tar.gz")
        print("  python setup_server_cache.py --verify runwayml/stable-diffusion-v1-5")
        print("  python setup_server_cache.py --list")
        print("  python setup_server_cache.py --setup-env")

if __name__ == "__main__":
    main()
