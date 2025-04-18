import torch
from torch.utils.data import DataLoader
from torch import autograd, nn

from transformers import BertModel, BertTokenizer
import re
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
from scipy.spatial import distance
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os
import collections
from model_def import TOXI_REG, DNN_module #, freeze
from toxi_dataloader import *
from toxi_dataloader import _get_train_data_loader, _get_test_data_loader #, freeze

def CCC(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

def lysis_metric_cal(y_true, y_pred, onehot_species_list, species_type, onehot_lysis_list, lysis_type):
    if species_type == 'Human':
        if lysis_type == 'HC50':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1. 0. 0. 0. 0. 0.]') & (str(onehot_lysis_list[i]) == '[1. 0. 0.]')]
        elif lysis_type == 'HC10':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1. 0. 0. 0. 0. 0.]') & (str(onehot_lysis_list[i]) == '[0. 1. 0.]')]
        elif lysis_type == 'HC5':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1. 0. 0. 0. 0. 0.]') & (str(onehot_lysis_list[i]) == '[0. 0. 1.]')]
    y_true_lysis = [y_true[i] for i in idx]
    y_pred_lysis = [y_pred[i] for i in idx]
    mse = mean_squared_error(y_true_lysis, y_pred_lysis)
    pcc= pearsonr(y_true_lysis, y_pred_lysis)[0]
    ktc = kendalltau(y_true_lysis, y_pred_lysis)[0]
    ccc = CCC(y_true_lysis, y_pred_lysis)
    r2 = r2_score(y_true_lysis, y_pred_lysis)
    return mse, pcc, ktc, ccc, r2

def csv_lysis_metric_cal(y_true, y_pred, onehot_species_list, species_type, onehot_lysis_list, lysis_type):
    if species_type == 'Human':
        if lysis_type == 'HC50':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1 0 0 0 0 0]') & (str(onehot_lysis_list[i]) == '[1 0 0]')]
        elif lysis_type == 'HC10':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1 0 0 0 0 0]') & (str(onehot_lysis_list[i]) == '[0 1 0]')]
        elif lysis_type == 'HC5':
            idx = [i for i in range(len(onehot_lysis_list)) if (str(onehot_species_list[i]) == '[1 0 0 0 0 0]') & (str(onehot_lysis_list[i]) == '[0 0 1]')]
    y_true_lysis = [y_true[i] for i in idx]
    y_pred_lysis = [y_pred[i] for i in idx]
    mse = mean_squared_error(y_true_lysis, y_pred_lysis)
    pcc= pearsonr(y_true_lysis, y_pred_lysis)[0]
    ktc = kendalltau(y_true_lysis, y_pred_lysis)[0]
    ccc = CCC(y_true_lysis, y_pred_lysis)
    r2 = r2_score(y_true_lysis, y_pred_lysis)
    return mse, pcc, ktc, ccc, r2

def csv_best_lysis_metric_record(best_target_list, best_predict_list, best_species_list, best_lysis_list):
    lysis_list, mse_list, pcc_list, ktc_list, ccc_list, r2_list = [], [], [], [], [], []
    hc_list = ['HC5', 'HC10', 'HC50']
    for hc in hc_list:
        lysis_list.append(hc)
        mse, pcc, ktc, ccc, r2 = csv_lysis_metric_cal(best_target_list, best_predict_list, best_species_list, 'Human', best_lysis_list, hc)
        mse_list.append(mse)
        pcc_list.append(pcc)
        ktc_list.append(ktc)
        ccc_list.append(ccc)
        r2_list.append(r2)
    return lysis_list, mse_list, pcc_list, ktc_list, ccc_list, r2_list

def best_lysis_metric_record(best_target_list, best_predict_list, best_species_list, best_lysis_list):
    lysis_list, mse_list, pcc_list, ktc_list, ccc_list, r2_list = [], [], [], [], [], []
    hc_list = ['HC5', 'HC10', 'HC50']
    for hc in hc_list:
        lysis_list.append(hc)
        mse, pcc, ktc, ccc, r2 = lysis_metric_cal(best_target_list, best_predict_list, best_species_list, 'Human', best_lysis_list, hc)
        mse_list.append(mse)
        pcc_list.append(pcc)
        ktc_list.append(ktc)
        ccc_list.append(ccc)
        r2_list.append(r2)
    return lysis_list, mse_list, pcc_list, ktc_list, ccc_list, r2_list

def fasta2csv(fasta_path, csv_path):
    f = open(fasta_path, "r")
    seq = collections.OrderedDict()
    for line in f:
        if line.startswith(">"):
            name = line.split()[0]
            name = name.replace(">", "")
            seq[name] = ''
        else:
            seq[name] += line.replace("\n", '').strip()
    f.close()
    ids_list = []
    sequences_list = []
    sequences_space_list = []
    lysis_list = []

    # For each sequence, create three rows with different lysis values
    lysis_values = ["HC5", "HC10", "HC50"]
    for seq_id, sequence in seq.items():
        for lysis in lysis_values:
            ids_list.append(seq_id)
            sequences_list.append(sequence)
            sequences_space_list.append(" ".join(sequence))
            lysis_list.append(lysis)

    seq_df = pd.DataFrame({
        'ID': ids_list,
        'SEQUENCE': sequences_list,
        'SEQUENCE_space': sequences_space_list,
        'lysis': lysis_list
    })
    seq_df['species'] = ['Human'] * len(seq_df)
    seq_df.to_csv(csv_path, index=False)
    return seq_df

def csv2fasta_func(csv_path, fasta_path):
    # get .csv info
    seq_data = pd.read_csv(csv_path)
    # .csv to .fasta
    fast_file = open(fasta_path, "w")
    for i in range(len(seq_data.SEQUENCE)):
        fast_file.write(">" + str(seq_data.ID[i]) + "\n")
        fast_file.write(seq_data.SEQUENCE[i] + "\n")
    fast_file.close()

def predict(pretrain_model_path, model_path, input_csv_path, output_csv_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device('mps')
    pretrain_bert = torch.load(pretrain_model_path)
    dnn = DNN_module(species_len=6, lysis_len=3, dropout=0.5)
    model = TOXI_REG(pretrain_bert=pretrain_bert, dnn=dnn)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model = model.to(device)

    test_df = pd.read_csv(input_csv_path)
    if 'SEQUENCE_space' not in test_df.columns:
        test_df['SEQUENCE_space'] = [" ".join(ele) for ele in test_df['SEQUENCE']]
        test_df.to_csv(input_csv_path, index=False)
    if 'pHC' not in test_df.columns:
        test_df['pHC'] = 0 * len(test_df)
        test_df.to_csv(input_csv_path, index=False)
    test_loader = _get_test_data_loader(500, input_csv_path)
    test_predict_list, test_target_list, onehot_species_list, onehot_lysis_list = [], [], [], []
    # seq_feature = [] #np.empty((0, 1024), dtype="float32")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets']
            b_species = batch['onehot_species'].to(device)
            b_lysis = batch['onehot_lysis'].to(device)
            predict_pHC, cls_embedding = model(input_ids=b_input_ids,
                                               attention_mask=b_input_mask,
                                               onehot_species=b_species,
                                               onehot_lysis=b_lysis)

            onehot_species_list.append(b_species.cpu().data.numpy())
            onehot_lysis_list.append(b_lysis.cpu().data.numpy())
            test_predict_list.append(predict_pHC.cpu().data.numpy())
            test_target_list.append(b_labels.data.numpy())

    ## store species and lysis for metric calculation
    # onehot_species_list = np.concatenate(onehot_species_list).flatten()
    # onehot_species_list = np.reshape(onehot_species_list, (-1, 6))  # six species [num_of_seq, 6]
    #
    # onehot_lysis_list = np.concatenate(onehot_lysis_list).flatten()
    # onehot_lysis_list = np.reshape(onehot_lysis_list, (-1, 3))  # five lysis value [num_of_seq, 3]
    test_predict_list = np.concatenate(test_predict_list).flatten()
    # test_target_list = np.concatenate(test_target_list).flatten()

    test_df['predicted_pHC'] = test_predict_list
    test_df["predicted_HC"] = [10 ** (-item) for item in test_predict_list]  # transform pHC value to HC
    test_df = test_df[['ID', 'SEQUENCE', 'species', 'lysis', 'predicted_HC']]
    test_df.to_csv(output_csv_path, index=False)
    return test_df


if __name__ == "__main__":
    # load the final DAPT+prefix+task_finetune model
    pretrain_model_path =  'toxic_pep_prot_bert.pth'
    model_path = 'proposed_model_reproduce.pkl'

    # load an input fasta file and store sequences in an input csv file
    input_fasta_path = 'example_input.fasta'
    input_csv_path = 'example_input.csv'
    fasta2csv(input_fasta_path, input_csv_path)

    # predict sequences, and store prediction result in an output csv file
    output_csv_path = 'example_output.csv'
    test_df = predict(pretrain_model_path, model_path, input_csv_path, output_csv_path)
    print("smart")