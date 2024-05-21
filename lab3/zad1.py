import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from dataclasses import dataclass
from collections import Counter


@dataclass
class Instance:
    text: list[str]
    label: str

class Vocab:
    def __init__(self, corpus, max_size=-1, min_freq=0, usespecialsigns=True):
        """
        corpus: iterable of strings
        sets self.itos and self,stoid
        """
        counts = Counter(corpus)
        filtered_counts = {word: count for word, count in counts.items() if count >= min_freq}
        sorted_strings = sorted(filtered_counts.keys(), key=lambda x: (-filtered_counts[x], x))
        mapping = {string: code for code, string in enumerate(sorted_strings, start=2)}
        if usespecialsigns:
            mapping['<PAD>'] = 0
            mapping['<UNK>']= 1
    
        if max_size >0 and len(mapping) > max_size:
            mapping = dict(list(mapping.items())[:max_size])

        self.stoi=mapping
        reversed_mapping = {v: k for k, v in mapping.items()}
        self.itos=reversed_mapping


    def encode(self, source):
        if isinstance(source, list):
            cleaned = [s if s in self.stoi.keys() else '<UNK>' for s in source]
            arr=[self.stoi[c] for c in cleaned]
            res=torch.tensor(arr)
        else:
            cleaned=source if source in self.stoi.keys() else '<UNK>'
            res=torch.tensor(self.stoi[cleaned])
        return res
    

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
        return self.text_vocab.encode(text), self.label_vocab.encode(label)

# Instantiate the datasets
#text_vocab=Vocab()
#train_dataset = NLPDataset('lab3/data/sst_train_raw.csv')
#valid_dataset = NLPDataset('lab3/data/sst_valid_raw.csv')
#test_dataset = NLPDataset('lab3/data/sst_test_raw.csv')

# Create data loaders
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop logic here
