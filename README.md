# BERT-HemoPep60-Predict-FastAPI

基於transformer架構和領域適應性預訓練(DAPT)的深度學習方法，用於定量預測肽對人類紅細胞的溶血活性的FastAPI微服務。

## 功能

- 預測單個肽序列的溶血活性
- 批量預測多個肽序列的溶血活性
- 支援FASTA格式的序列輸入

## API 端點

- `/api/predict`: 預測肽序列的溶血活性（支援單個或多個序列）
- `/api/docs`: API文檔

## 技術棧

- FastAPI: 高性能的現代化API框架
- PyTorch: 深度學習模型的實現
- Transformers: 使用BERT模型處理肽序列
- Docker: 容器化部署
- Docker Compose: 容器編排

## 本地開發與測試

### 使用Docker（推薦）

1. 克隆存儲庫:
   ```
   git clone https://github.com/yourusername/BERT-HemoPep60-Predict-FastAPI.git
   cd BERT-HemoPep60-Predict-FastAPI
   ```

2. 下載模型文件並放入models目錄:
   ```
   mkdir -p models
   # 將以下兩個文件放入models目錄
   # - proposed_model_reproduce.pkl
   # - toxic_pep_prot_bert.pth
   ```

3. 構建並啟動Docker容器:
   ```
   docker compose build
   docker compose up -d
   ```

4. 測試API:
   ```
   curl -X POST "http://localhost:9001/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "fasta": ">Peptide1\nAAKIILNPKFR",
       "model_type": "BERT-HemoPep60"
     }'
   ```

5. 訪問API文檔:
   ```
   http://localhost:9001/api/docs
   ```

### 不使用Docker

1. 克隆存儲庫:
   ```
   git clone https://github.com/yourusername/BERT-HemoPep60-Predict-FastAPI.git
   cd BERT-HemoPep60-Predict-FastAPI
   ```

2. 安裝依賴:
   ```
   pip install -r requirements.txt
   ```

3. 下載模型文件並放入models目錄

4. 運行服務:
   ```
   uvicorn api.index:app --host 0.0.0.0 --port 9001
   ```

## 部署

### 使用Docker部署到伺服器

1. 在伺服器上安裝Docker和Docker Compose

2. 將代碼上傳到伺服器:
   ```
   rsync -avz --exclude 'models/' --exclude '.git/' --exclude '__pycache__/' --exclude '.env' . user@your-server:/path/to/deployment/
   ```

3. 上傳模型文件:
   ```
   # 確保伺服器上有models目錄
   ssh user@your-server "mkdir -p /path/to/deployment/models"
   
   # 上傳模型文件
   scp models/proposed_model_reproduce.pkl user@your-server:/path/to/deployment/models/
   scp models/toxic_pep_prot_bert.pth user@your-server:/path/to/deployment/models/
   ```

4. 在伺服器上構建並啟動容器:
   ```
   ssh user@your-server "cd /path/to/deployment && docker compose build && docker compose up -d"
   ```

## API 使用範例

### 單一序列預測

```bash
curl -X POST "http://your-server:9001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fasta": ">Melittin\nGIGAVLKVLTTGLPALISWIKRKRQQ",
    "model_type": "BERT-HemoPep60"
  }'
```

### 多序列批量預測

```bash
curl -X POST "http://your-server:9001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fasta": ">Peptide1\nAAKIILNPKFR\n>Peptide2\nKWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQAATIYK",
    "model_type": "BERT-HemoPep60"
  }'
```

## 結果解讀

- **HC5/HC10/HC50**: 這些值表示需要多少濃度（μM）的肽段才能引起5%/10%/50%的紅血球溶解
- **較低的值**: 表示較強的溶血活性（毒性更高）
- **較高的值**: 表示較弱的溶血活性（毒性更低）

## 模型信息

BERT-HemoPep60是一種基於transformer架構和領域適應性預訓練(DAPT)的深度學習方法，用於定量預測肽對人類紅細胞的溶血活性。模型使用創新的前綴提示方法整合了六種常見哺乳動物物種（人類、小鼠、大鼠、馬、羊、兔）的實驗溶血數據和多種溶血指標（HC5、HC10、HC50）進行肽毒性預測。

## 引用

如果您使用了此API，請考慮引用原始研究:

```
BERT-HemoPep60: A Deep Learning Method for Quantitative Hemolytic Activity Prediction of Peptides Based on Transformers and Domain Adaptive Pretraining
``` 