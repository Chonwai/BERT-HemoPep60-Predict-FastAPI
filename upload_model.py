from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv()

# 使用你的API Token登錄
login(os.getenv("HUGGINGFACE_TOKEN"))

# 創建API實例
api = HfApi()

# 創建或使用現有的模型倉庫
repo_id = "Edison/BERT-HemoPep60-Predict"
try:
    api.create_repo(repo_id=repo_id, private=True)
    print(f"Created repository: {repo_id}")
except Exception as e:
    print(f"Repository already exists or error: {str(e)}")

# 上傳模型文件 - 主模型
print("上傳proposed_model_reproduce.pkl... (這可能需要一些時間)")
api.upload_file(
    path_or_fileobj="./proposed_model_reproduce.pkl",
    path_in_repo="proposed_model_reproduce.pkl",
    repo_id=repo_id
)
print("主模型上傳完成！")

# 上傳預訓練模型文件
print("上傳toxic_pep_prot_bert.pth... (這可能需要更長時間)")
api.upload_file(
    path_or_fileobj="./toxic_pep_prot_bert.pth",
    path_in_repo="toxic_pep_prot_bert.pth",
    repo_id=repo_id
)
print("預訓練模型上傳完成！")
