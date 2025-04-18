import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

d_model = 1024
batch_size = 8
MAX_LEN = 70
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )

def __Match(x, groups):
    try:
        clas = [y for y in groups if y in x][0]
    except:
        clas = 'other'
    return clas

def __GetOneHotPrefix(csv_path):
    df = pd.read_csv(csv_path)
    # Builds one hot encoded numpy arrays for given species
    list_of_species = ['Human', 'Mouse', 'Sheep', 'Rabbit', 'Horse', 'Rat']
    list_of_species_tmp = list_of_species.copy()
    if len(list_of_species_tmp) > 1:
        species_hot_enc_dict = dict(zip(list_of_species_tmp, np.eye(len(list_of_species_tmp), dtype=int).tolist()))
        df = df.reset_index().drop(columns='index', axis=1)
        species_clas = df.species.apply(lambda x: __Match(x, list_of_species_tmp)) # species string type to onehot
        species_encoded_clas = species_clas.apply(lambda x: np.array(species_hot_enc_dict[x]))
        df['onehot_species'] = species_encoded_clas

    # Builds one hot encoded numpy arrays for given lysiss.
    list_of_lysis = ['HC50', 'HC10', 'HC5']
    list_of_lysis_tmp = list_of_lysis.copy()
    if len(list_of_lysis_tmp) > 1:
        lysis_hot_enc_dict = dict(zip(list_of_lysis_tmp, np.eye(len(list_of_lysis_tmp), dtype=int).tolist()))
        df = df.reset_index().drop(columns='index', axis=1)
        lysis_clas = df.lysis.apply(lambda x: __Match(x, list_of_lysis_tmp)) # lysis string type to onehot
        lysis_encoded_clas = lysis_clas.apply(lambda x: np.array(lysis_hot_enc_dict[x]))
        df['onehot_lysis'] = lysis_encoded_clas
    return df

# my_train = __GetOneHotPrefix('data/0/train.csv')

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


def _get_train_data_loader(batch_size, train_dir, train_frac):
    # dataset = pd.read_csv(train_dir)
    dataset = __GetOneHotPrefix(train_dir)
    dataset = dataset.sample(frac = train_frac)
    train_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        onehot_species=dataset.onehot_species.to_numpy(),
        onehot_lysis = dataset.onehot_lysis.to_numpy(),
        targets=dataset.pHC.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_dataloader

def _get_test_data_loader(batch_size, test_dir):
    # dataset = pd.read_csv(test_dir)
    dataset = __GetOneHotPrefix(test_dir)
    test_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        onehot_species=dataset.onehot_species.to_numpy(),
        onehot_lysis=dataset.onehot_lysis.to_numpy(),
        targets=dataset.pHC.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_dataloader
