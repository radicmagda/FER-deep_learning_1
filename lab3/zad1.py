import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from dataclasses import dataclass


@dataclass
class Instance:
    text: list[str]
    label: str

class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.vf=frequencies

    def encode(source):
        #must return a tensor of numerical values of words
        return torch.tensor(0)

class NLPDataset(Dataset):
    def __init__(self, csv_file, text_vocab:Vocab, label_vocab:Vocab):
        data_frame = pd.read_csv(csv_file)
        self.instances=[Instance(text.split(), label) for text, label in data_frame.values] #lista Instance-ova

        self.text_vocab=text_vocab
        self.label_vocab=label_vocab


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        #vraca: numerikalizirani tekst (torch.tensor 1.reda/listu), numerikaliziranu labelu (torch.tensor 0.reda/skalar)
        inst=self.instances[idx]
        text=inst.text
        label=inst.label
        return self.text_vocab.encode(text), self.

# Instantiate the datasets
train_dataset = NLPDataset('lab3/data/sst_train_raw.csv')
valid_dataset = NLPDataset('lab3/data/sst_valid_raw.csv')
test_dataset = NLPDataset('lab3/data/sst_test_raw.csv')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example of iterating through the data loaders
for batch in train_loader:
    features, labels = batch['features'], batch['labels']
    print(features, labels)
    # Training loop logic here
