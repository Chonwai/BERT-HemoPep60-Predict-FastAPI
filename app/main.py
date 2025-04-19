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
from dotenv import load_dotenv


from .model.inference import load_model, predict_peptide_hemolytic_activity

# 加載環境變量
load_dotenv()

description = """
# BERT-HemoPep60 Prediction API

Based on the transformer architecture and domain-adaptive pre-training (DAPT), this deep learning method is used for quantitative prediction of peptide hemolytic activity against human red blood cells.

## Features

* `/api/predict`: Predict hemolytic activity of peptide sequences (supports FASTA format input)
"""

app = FastAPI(
    title="BERT-HemoPep60 API",
    description=description,
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model, pretrain_model = None, None

@app.on_event("startup")
async def startup_event():
    global model, pretrain_model
    try:
        model, pretrain_model = load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())

class PredictRequest(BaseModel):
    fasta: str
    model_type: str = "BERT-HemoPep60"  # Default model type

@app.get("/")
async def root():
    return {"message": "BERT-HemoPep60 API is running!"}

@app.post("/api/predict")
async def predict_peptides(request: PredictRequest):
    """
    Predict hemolytic activity of peptide sequences using FASTA format input
    
    Input format:
    ```
    >sequence_id1
    PEPTIDESEQUENCE1
    >sequence_id2
    PEPTIDESEQUENCE2
    ```
    """
    global model, pretrain_model
    
    if model is None or pretrain_model is None:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": "Model not correctly loaded. Please try again later."
            }
        )
    
    try:
        # Parse FASTA format
        fasta_content = request.fasta.strip()
        
        # Check model_type (Optional feature, currently only supports one model)
        if request.model_type != "BERT-HemoPep60":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error", 
                    "detail": f"Unsupported model type: {request.model_type}. Currently only supporting 'BERT-HemoPep60'"
                }
            )
        
        # Parse FASTA content
        sequences = []
        sequence_ids = []
        
        lines = fasta_content.split('\n')
        current_id = None
        current_seq = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence (if any)
                if current_id is not None and current_seq:
                    sequence_ids.append(current_id)
                    sequences.append(current_seq)
                    current_seq = ""
                
                # Start new sequence
                current_id = line[1:]  # Remove > symbol
            else:
                current_seq += line
        
        # Save last sequence
        if current_id is not None and current_seq:
            sequence_ids.append(current_id)
            sequences.append(current_seq)
        
        # Check if there are valid sequences
        if not sequences:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": "No valid peptide sequences found. Please ensure FASTA format is correct."
                }
            )
        
        # Check sequence length
        invalid_seqs = [seq for seq in sequences if len(seq) < 5]
        if invalid_seqs:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": "Invalid peptide sequence. Sequences must be at least 5 amino acids long."
                }
            )
        
        # Predict
        predictions_result = predict_peptide_hemolytic_activity(model, pretrain_model, sequences)
        
        # Format response
        # Extract HC50 as the primary prediction value from the original predictions
        prediction_values = [float(pred["HC50"]) for pred in predictions_result]
        
        # Return uniform format response
        return {
            "status": "success",
            "data": {
                "fasta_ids": sequence_ids,
                "sequences": sequences,
                "predictions": prediction_values,
                "detailed_predictions": [
                    {
                        "sequence_id": seq_id,
                        "sequence": seq,
                        "HC5": float(predictions_result[i]["HC5"]),
                        "HC10": float(predictions_result[i]["HC10"]),
                        "HC50": float(predictions_result[i]["HC50"])
                    }
                    for i, (seq_id, seq) in enumerate(zip(sequence_ids, sequences))
                ]
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Error occurred during prediction: {str(e)}"
            }
        )

# Keep existing endpoints but mark them as deprecated
@app.post("/api/predict/", deprecated=True)
async def predict_peptide_deprecated(request: PredictRequest):
    """This endpoint is deprecated, please use /api/predict"""
    return await predict_peptides(request)

@app.post("/api/predict/batch/", deprecated=True)
async def predict_peptides_batch_deprecated(request: PredictRequest):
    """This endpoint is deprecated, please use /api/predict"""
    return await predict_peptides(request)

@app.post("/api/predict/fasta/", deprecated=True)
async def predict_from_fasta_deprecated(file: UploadFile = File(...)):
    """This endpoint is deprecated, please use /api/predict"""
    # Read file content
    content = await file.read()
    content = content.decode("utf-8")
    
    # Create mock request object
    mock_request = PredictRequest(fasta=content, model_type="BERT-HemoPep60")
    
    # Call new unified endpoint
    return await predict_peptides(mock_request) 