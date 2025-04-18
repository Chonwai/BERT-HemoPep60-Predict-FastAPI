import torch
import torch.nn as nn
import numpy as np

class DNN_module(nn.Module):
    # nn.Module class to generate DNN module as a transformer regression head.
    def __init__(self, species_len, lysis_len, dropout):
        super(DNN_module, self).__init__()
        self.species_len = species_len,
        self.lysis_len = lysis_len,
        self.dropout = nn.Dropout(dropout)
        self.active = nn.ReLU()

        self.fc1 = nn.Linear(1024 + species_len + lysis_len, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.fc3(x)
        return x

class TOXI_REG(nn.Module):
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


# def freeze(model, frozen_layers):
#     modules = [model.pretrain_bert.encoder.layer[:frozen_layers]]
#     for module in modules:
#         for param in module.parameters():
#             param.requires_grad = False
