# BERT-HemoPep60-Predict-FastAPI

基於transformer架構和領域適應性預訓練(DAPT)的深度學習方法，用於定量預測肽對人類紅細胞的溶血活性的FastAPI微服務。

## 功能

- 預測單個肽序列的溶血活性
- 批量預測多個肽序列的溶血活性
- 上傳FASTA文件進行預測

## API 端點

- `/api/predict/`: 預測單個肽序列的溶血活性
- `/api/predict/batch/`: 批量預測多個肽序列的溶血活性
- `/api/predict/fasta/`: 上傳FASTA文件進行預測
- `/api/docs`: API文檔

## 技術棧

- FastAPI: 高性能的現代化API框架
- PyTorch: 深度學習模型的實現
- Transformers: 使用BERT模型處理肽序列
- Vercel: 無服務器部署平台

## 本地開發

1. 克隆存儲庫:
   ```
   git clone https://github.com/yourusername/BERT-HemoPep60-Predict-FastAPI.git
   cd BERT-HemoPep60-Predict-FastAPI
   ```

2. 安裝依賴:
   ```
   pip install -r requirements.txt
   ```

3. 運行服務:
   ```
   uvicorn api.index:app --reload
   ```

4. 訪問本地API文檔:
   ```
   http://localhost:8000/api/docs
   ```

## 部署

此應用程序配置為在Vercel上部署。只需將存儲庫連接到Vercel帳戶，它就會自動部署。

## 模型信息

BERT-HemoPep60是一種基於transformer架構和領域適應性預訓練(DAPT)的深度學習方法，用於定量預測肽對人類紅細胞的溶血活性。模型使用創新的前綴提示方法整合了六種常見哺乳動物物種（人類、小鼠、大鼠、馬、羊、兔）的實驗溶血數據和多種溶血指標（HC5、HC10、HC50）進行肽毒性預測。

## 引用

如果您使用了此API，請考慮引用原始研究:

```
BERT-HemoPep60: A Deep Learning Method for Quantitative Hemolytic Activity Prediction of Peptides Based on Transformers and Domain Adaptive Pretraining
``` 