from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import torch
import numpy as np
import pandas as pd
import collections
import traceback
import json
from io import StringIO


from .model.inference import load_model, predict_peptide_hemolytic_activity

description = """
# BERT-HemoPep60 預測 API

基於transformer架構和領域適應性預訓練(DAPT)的深度學習方法，用於定量預測肽對人類紅細胞的溶血活性。

## 功能

* `/predict/`: 預測單個肽序列的溶血活性
* `/predict/batch/`: 批量預測多個肽序列的溶血活性
* `/predict/fasta/`: 上傳FASTA文件進行預測
"""

app = FastAPI(
    title="BERT-HemoPep60 API",
    description=description,
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加載模型
model, pretrain_model = None, None

@app.on_event("startup")
async def startup_event():
    global model, pretrain_model
    try:
        model, pretrain_model = load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())

class PeptideRequest(BaseModel):
    sequence: str
    
class BatchPeptideRequest(BaseModel):
    sequences: List[str]

class PeptideResponse(BaseModel):
    sequence: str
    predicted_HC5: float
    predicted_HC10: float
    predicted_HC50: float
    
@app.get("/")
async def root():
    return {"message": "BERT-HemoPep60 API is running!"}

@app.post("/api/predict/", response_model=PeptideResponse)
async def predict_peptide(request: PeptideRequest):
    global model, pretrain_model
    
    if model is None or pretrain_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        sequence = request.sequence.strip()
        if not sequence or len(sequence) < 5:
            raise HTTPException(status_code=400, detail="Invalid peptide sequence. Must be at least 5 amino acids.")
        
        results = predict_peptide_hemolytic_activity(model, pretrain_model, [sequence])
        
        # 返回預測結果
        return {
            "sequence": sequence,
            "predicted_HC5": float(results[0]["HC5"]),
            "predicted_HC10": float(results[0]["HC10"]),
            "predicted_HC50": float(results[0]["HC50"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/batch/", response_model=List[PeptideResponse])
async def predict_peptides_batch(request: BatchPeptideRequest):
    global model, pretrain_model
    
    if model is None or pretrain_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        sequences = [seq.strip() for seq in request.sequences if seq.strip()]
        if not sequences:
            raise HTTPException(status_code=400, detail="No valid peptide sequences provided.")
        
        results = predict_peptide_hemolytic_activity(model, pretrain_model, sequences)
        
        # 返回預測結果
        return [
            {
                "sequence": seq,
                "predicted_HC5": float(results[i]["HC5"]),
                "predicted_HC10": float(results[i]["HC10"]),
                "predicted_HC50": float(results[i]["HC50"])
            }
            for i, seq in enumerate(sequences)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/fasta/", response_model=List[PeptideResponse])
async def predict_from_fasta(file: UploadFile = File(...)):
    global model, pretrain_model
    
    if model is None or pretrain_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        # 解析FASTA文件
        content = await file.read()
        content = content.decode("utf-8")
        
        seq = collections.OrderedDict()
        for line in content.split("\n"):
            if line.startswith(">"):
                name = line.split()[0]
                name = name.replace(">", "")
                seq[name] = ''
            else:
                seq[name] = seq.get(name, '') + line.strip()
        
        sequences = list(seq.values())
        if not sequences:
            raise HTTPException(status_code=400, detail="No valid peptide sequences found in FASTA file.")
        
        results = predict_peptide_hemolytic_activity(model, pretrain_model, sequences)
        
        # 返回預測結果
        response_data = []
        for i, (seq_id, sequence) in enumerate(seq.items()):
            if i < len(results):
                response_data.append({
                    "sequence": sequence,
                    "predicted_HC5": float(results[i]["HC5"]),
                    "predicted_HC10": float(results[i]["HC10"]),
                    "predicted_HC50": float(results[i]["HC50"])
                })
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FASTA processing error: {str(e)}") 