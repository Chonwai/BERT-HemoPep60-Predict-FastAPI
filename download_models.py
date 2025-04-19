#!/usr/bin/env python3
"""
下载模型文件到本地目录
预先下载大型模型文件并存储在本地，避免每次启动容器时都要下载
"""

import os
import sys
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Edison/BERT-HemoPep60-Predict")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./models")

def download_model(filename):
    """从Hugging Face下载模型到本地目录"""
    print(f"Downloading {filename} from {HF_MODEL_ID}...")
    
    try:
        # 确保本地目录存在
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        
        # 下载文件
        path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename=filename,
            token=HF_TOKEN
        )
        
        # 复制到本地目录
        local_file = os.path.join(LOCAL_MODEL_PATH, filename)
        if not os.path.exists(local_file):
            print(f"Copying {filename} to {local_file}")
            import shutil
            shutil.copy(path, local_file)
            print(f"Successfully saved {filename} to {local_file}")
        else:
            print(f"File already exists at {local_file}")
        
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    """主函数"""
    # 如果提供了token，先登录
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    # 下载模型文件
    models = [
        "proposed_model_reproduce.pkl",
        "toxic_pep_prot_bert.pth"
    ]
    
    success = True
    for model in models:
        if not download_model(model):
            success = False
    
    if success:
        print("All models downloaded successfully!")
        return 0
    else:
        print("Failed to download some models.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 