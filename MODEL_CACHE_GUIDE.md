# 模型缓存管理指南

本指南帮助你管理 Hugging Face 模型缓存，特别是在 Windows 本地下载后上传到 Linux 服务器的场景。

## 📁 缓存位置

### Windows 系统
```
C:\Users\{用户名}\.cache\huggingface\hub\
```

### Linux 系统
```
~/.cache/huggingface/hub/
```

### 环境变量控制
- `HF_HOME`: Hugging Face 主目录
- `HUGGINGFACE_HUB_CACHE`: 直接指定缓存目录

## 🔍 查找本地缓存

### 1. 使用提供的脚本
```bash
# 查看缓存目录位置
python check_model_cache.py --cache_dir

# 查看特定模型
python check_model_cache.py --model_id runwayml/stable-diffusion-v1-5

# 列出所有缓存的模型
python check_model_cache.py --list_all
```

### 2. 手动查找
模型缓存目录命名规则：
```
models--{组织名}--{模型名}
```

例如：
- `runwayml/stable-diffusion-v1-5` → `models--runwayml--stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1-base` → `models--stabilityai--stable-diffusion-2-1-base`

## 📦 打包和上传

### 1. 在 Windows 上打包
```bash
# 进入缓存目录
cd C:\Users\{用户名}\.cache\huggingface\hub\

# 打包特定模型
tar -czf stable-diffusion-v1-5.tar.gz models--runwayml--stable-diffusion-v1-5

# 或者打包所有模型
tar -czf all-models.tar.gz models--*
```

### 2. 上传到服务器
```bash
# 使用 scp
scp stable-diffusion-v1-5.tar.gz user@server:/tmp/

# 使用 rsync
rsync -avz stable-diffusion-v1-5.tar.gz user@server:/tmp/
```

### 3. 在服务器上解压
```bash
# 创建缓存目录
mkdir -p ~/.cache/huggingface/hub

# 解压到缓存目录
cd ~/.cache/huggingface/hub
tar -xzf /tmp/stable-diffusion-v1-5.tar.gz

# 验证
ls -la models--runwayml--stable-diffusion-v1-5
```

## 🛠️ 使用服务器脚本

### 1. 上传脚本到服务器
```bash
scp setup_server_cache.py user@server:/tmp/
```

### 2. 在服务器上使用
```bash
# 解压模型缓存
python setup_server_cache.py --extract /tmp/stable-diffusion-v1-5.tar.gz

# 验证模型缓存
python setup_server_cache.py --verify runwayml/stable-diffusion-v1-5

# 自定义缓存目录
python setup_server_cache.py --extract /tmp/stable-diffusion-v1-5.tar.gz --cache-dir /data/hf_cache
```

## 📋 常见模型及其缓存大小

| 模型 | 缓存目录名 | 大小 (约) |
|------|------------|-----------|
| runwayml/stable-diffusion-v1-5 | models--runwayml--stable-diffusion-v1-5 | ~4.2GB |
| stabilityai/stable-diffusion-2-1-base | models--stabilityai--stable-diffusion-2-1-base | ~5.1GB |
| openai/clip-vit-large-patch14 | models--openai--clip-vit-large-patch14 | ~1.7GB |

## 🔧 环境变量设置

### 临时设置
```bash
export HF_HOME=/data/huggingface
export HUGGINGFACE_HUB_CACHE=/data/huggingface/hub
```

### 永久设置
```bash
# 添加到 ~/.bashrc
echo 'export HF_HOME=/data/huggingface' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/data/huggingface/hub' >> ~/.bashrc
source ~/.bashrc
```

## 🚨 注意事项

1. **磁盘空间**: 确保服务器有足够空间存储模型文件
2. **权限**: 确保对缓存目录有读写权限
3. **网络**: 如果缓存不完整，脚本可能仍会尝试下载缺失文件
4. **版本**: 确保本地和服务器使用相同版本的 transformers/diffusers

## 🔍 故障排除

### 问题1: 模型仍在下载
**原因**: 缓存不完整或路径不正确
**解决**: 
```bash
# 检查缓存完整性
python setup_server_cache.py --verify runwayml/stable-diffusion-v1-5

# 检查环境变量
echo $HUGGINGFACE_HUB_CACHE
```

### 问题2: 权限错误
**原因**: 缓存目录权限不足
**解决**:
```bash
# 修改权限
chmod -R 755 ~/.cache/huggingface/
```

### 问题3: 空间不足
**原因**: 磁盘空间不够
**解决**:
```bash
# 检查空间
df -h ~/.cache/huggingface/

# 清理旧缓存
rm -rf ~/.cache/huggingface/hub/models--old-model-name
```

## 📝 快速命令参考

```bash
# 本地检查
python check_model_cache.py --model_id runwayml/stable-diffusion-v1-5

# 打包上传
tar -czf model.tar.gz models--runwayml--stable-diffusion-v1-5
scp model.tar.gz server:/tmp/

# 服务器解压
python setup_server_cache.py --extract /tmp/model.tar.gz

# 验证
python setup_server_cache.py --verify runwayml/stable-diffusion-v1-5
```
