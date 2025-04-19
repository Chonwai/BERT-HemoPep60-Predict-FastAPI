FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖和Rust工具链
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH"

# 设置Rust环境变量
ENV PATH="/root/.cargo/bin:${PATH}"

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖，使用CPU版本的PyTorch以减少镜像大小
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    # 预下载transformers模型避免运行时下载
    python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('Rostlab/prot_bert'); BertModel.from_pretrained('Rostlab/prot_bert')"

# 创建模型目录
RUN mkdir -p /app/models

# 首先只复制下载模型和启动脚本
COPY download_models.py /app/
COPY start.sh /app/
RUN chmod +x /app/start.sh

# 复制其余应用代码
COPY . .

# 指定环境变量
ENV PYTHONPATH=/app
ENV LOCAL_MODEL_PATH=/app/models

# 曝露端口
EXPOSE 9001

# 使用启动脚本
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "9001"] 