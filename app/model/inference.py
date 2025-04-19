import os
import torch
import numpy as np
import pandas as pd
import collections
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import threading
import concurrent.futures
import time

# 模型定義
class DNN_module(torch.nn.Module):
    # nn.Module class to generate DNN module as a transformer regression head.
    def __init__(self, species_len, lysis_len, dropout):
        super(DNN_module, self).__init__()
        self.species_len = species_len,
        self.lysis_len = lysis_len,
        self.dropout = torch.nn.Dropout(dropout)
        self.active = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(1024 + species_len + lysis_len, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 1)
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.fc3(x)
        return x

class TOXI_REG(torch.nn.Module):
    def __init__(self, pretrain_bert, dnn):
        super(TOXI_REG, self).__init__()
        self.pretrain = pretrain_bert
        self.dnn = dnn
    def forward(self, input_ids, attention_mask, onehot_species, onehot_lysis):
        pretrain_output = self.pretrain(input_ids, attention_mask)[0]
        pretrain_cls = pretrain_output[:, 0, :]
        inputs = torch.cat((pretrain_cls, onehot_species, onehot_lysis), 1)
        out = self.dnn(inputs)
        return out, pretrain_cls

# 資料加載和處理
class Seq_Dataset(Dataset):
    def __init__(self, sequence, onehot_species, onehot_lysis, targets, tokenizer, max_len):
        self.sequence = sequence
        self.onehot_species = onehot_species
        self.onehot_lysis = onehot_lysis
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        onehot_species = self.onehot_species[item]
        onehot_lysis = self.onehot_lysis[item]
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'protein_sequence': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'onehot_species': torch.tensor(onehot_species, dtype=torch.float),
            'onehot_lysis': torch.tensor(onehot_lysis, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.float)
        }

def _get_test_data_loader(batch_size, sequences, onehot_species_list, onehot_lysis_list, targets, tokenizer, max_len=70):
    test_data = Seq_Dataset(
        sequence=sequences,
        onehot_species=onehot_species_list,
        onehot_lysis=onehot_lysis_list,
        targets=targets,
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_dataloader

def load_model():
    """加載預訓練模型並返回用於預測的模型"""
    import time
    
    # 獲取HF配置
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Edison/BERT-HemoPep60-Predict")
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
    
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    
    # 定義獲取模型的函數，優先從本地路徑加載
    def get_model_path(filename):
        # 檢查本地路徑
        if LOCAL_MODEL_PATH:
            local_file = os.path.join(LOCAL_MODEL_PATH, filename)
            if os.path.exists(local_file):
                print(f"Using local model file: {local_file}")
                return local_file
        
        # 本地文件不存在，從Hugging Face下載
        print(f"Local model file not found, downloading {filename} from Hugging Face...")
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_ID,
                filename=filename,
                token=HF_TOKEN
            )
            print(f"Download of {filename} completed successfully!")
            
            # 如果指定了本地模型路徑，將下載的模型複製到那裡
            if LOCAL_MODEL_PATH:
                os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
                local_file = os.path.join(LOCAL_MODEL_PATH, filename)
                if not os.path.exists(local_file):
                    print(f"Copying downloaded model to {local_file}")
                    import shutil
                    shutil.copy(path, local_file)
            
            return path
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return None
    
    # 1. 首先嘗試獲取主模型權重
    print("Getting main model weights...")
    try:
        model_weights_path = get_model_path("proposed_model_reproduce.pkl")
        if not model_weights_path:
            raise Exception("Failed to obtain main model weights, cannot continue.")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise
    
    # 2. 然後嘗試獲取預訓練模型
    print("Getting pretrained model...")
    try:
        pretrain_model_path = get_model_path("toxic_pep_prot_bert.pth")
    except Exception as e:
        print(f"Warning: Could not obtain pretrained model: {str(e)}")
        pretrain_model_path = None
    
    # 加載預訓練模型
    if pretrain_model_path:
        print(f"Loading pretrained model from {pretrain_model_path}...")
        try:
            from transformers import BertModel
            pretrain_bert_base = BertModel.from_pretrained("Rostlab/prot_bert")
            
            # 使用pretrained權重替換基本模型的狀態
            pretrain_state_dict = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
            # 檢查加載的物件類型
            if isinstance(pretrain_state_dict, torch.nn.Module):
                # 如果是模型實例，獲取其狀態字典
                pretrain_state_dict = pretrain_state_dict.state_dict()
                
            # 嘗試加載預訓練模型權重
            pretrain_bert_base.load_state_dict(pretrain_state_dict, strict=False)
            pretrain_bert = pretrain_bert_base
            print("Custom pretrained model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading custom pretrained model, falling back to Rostlab/prot_bert: {str(e)}")
            # 如果加載失敗，使用標準預訓練模型
            from transformers import BertModel
            pretrain_bert = BertModel.from_pretrained("Rostlab/prot_bert")
    else:
        print("Using default pretrained model from Rostlab/prot_bert")
        from transformers import BertModel
        pretrain_bert = BertModel.from_pretrained("Rostlab/prot_bert")
    
    # 初始化模型
    print("Initializing model...")
    dnn = DNN_module(species_len=6, lysis_len=3, dropout=0.5)
    model = TOXI_REG(pretrain_bert=pretrain_bert, dnn=dnn)
    
    # 加載模型權重
    print(f"Loading model weights from {model_weights_path}...")
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')), strict=False)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
    
    # 設定為評估模式
    model.eval()
    
    return model, pretrain_bert

def predict_peptide_hemolytic_activity(model, pretrain_model, sequences):
    """預測肽序列的溶血活性
    
    Args:
        model: 已加載的模型
        pretrain_model: 預訓練模型
        sequences: 肽序列列表
        
    Returns:
        溶血活性預測結果列表
    """
    device = torch.device('cpu')
    model = model.to(device)
    
    # 使用Hugging Face的tokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    
    # 準備數據
    # 添加空格到序列中以適應Prot_BERT的輸入格式
    sequences_spaced = [" ".join(seq) for seq in sequences]
    
    # 創建三種溶血活性測量值的數據
    all_sequences = []
    all_onehot_species = []
    all_onehot_lysis = []
    all_targets = []
    
    lysis_values = ["HC5", "HC10", "HC50"]
    lysis_onehot = {
        "HC5": [0, 0, 1],
        "HC10": [0, 1, 0],
        "HC50": [1, 0, 0]
    }
    
    # 人類種類的one-hot編碼
    human_species_onehot = [1, 0, 0, 0, 0, 0]
    
    # 為每個序列創建三個輸入（一個用於每種溶血度量）
    for seq in sequences_spaced:
        for lysis in lysis_values:
            all_sequences.append(seq)
            all_onehot_species.append(human_species_onehot)
            all_onehot_lysis.append(lysis_onehot[lysis])
            all_targets.append(0)  # 目標值在預測中不重要
    
    # 創建數據加載器
    batch_size = min(len(all_sequences), 32)  # 批處理大小，根據可用內存調整
    test_loader = _get_test_data_loader(
        batch_size,
        all_sequences,
        all_onehot_species,
        all_onehot_lysis,
        all_targets,
        tokenizer
    )
    
    # 預測
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_species = batch['onehot_species'].to(device)
            b_lysis = batch['onehot_lysis'].to(device)
            
            predict_pHC, _ = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                onehot_species=b_species,
                onehot_lysis=b_lysis
            )
            
            predictions.extend(predict_pHC.cpu().data.numpy().flatten())
    
    # 變換pHC預測為HC值
    hc_predictions = [10 ** (-float(pHC)) for pHC in predictions]
    
    # 重組結果
    results = []
    for i in range(0, len(hc_predictions), 3):
        if i + 2 < len(hc_predictions):
            results.append({
                "HC5": hc_predictions[i + 2],
                "HC10": hc_predictions[i + 1],
                "HC50": hc_predictions[i]
            })
    
    return results 