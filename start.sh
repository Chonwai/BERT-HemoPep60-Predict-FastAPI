#!/bin/bash
set -e

# 检查并创建模型目录
MODEL_DIR=${LOCAL_MODEL_PATH:-/app/models}
mkdir -p $MODEL_DIR

# 预下载模型文件(如果不存在)
if [ -f "$MODEL_DIR/proposed_model_reproduce.pkl" ] && [ -f "$MODEL_DIR/toxic_pep_prot_bert.pth" ]; then
    echo "Models already exist in $MODEL_DIR"
else
    echo "Downloading models to $MODEL_DIR..."
    python download_models.py
fi

# 启动FastAPI应用
echo "Starting FastAPI application..."
exec uvicorn api.index:app --host 0.0.0.0 --port 8000 --workers 1 