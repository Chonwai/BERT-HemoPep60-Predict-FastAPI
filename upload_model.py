from huggingface_hub import HfApi, login
import os

# 使用你的API Token登錄
login("你的_HF_TOKEN")

# 創建API實例
api = HfApi()

# 創建新的模型倉庫
repo_id = "你的用戶名/BERT-HemoPep60"
api.create_repo(repo_id=repo_id, private=True)  # 設置private=False如果你想公開模型

# 上傳模型文件
print("上傳proposed_model_reproduce.pkl... (這可能需要一些時間)")
api.upload_file(
    path_or_fileobj="./proposed_model_reproduce.pkl",
    path_in_repo="proposed_model_reproduce.pkl",
    repo_id=repo_id
)

print("模型上傳完成！")
