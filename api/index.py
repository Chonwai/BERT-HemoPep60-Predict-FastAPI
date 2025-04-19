from app.main import app
import os
from dotenv import load_dotenv

# 加載環境變量（在Vercel上，這些將從Vercel項目設置中獲取）
load_dotenv()

# 使用Vercel的入口點格式導出FastAPI應用
# 這個文件是Vercel部署的入口點 