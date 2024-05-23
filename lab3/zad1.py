import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from dataclasses import dataclass
from collections import Counter
from torch import nn

TRAIN_PATH='lab3/data/sst_train_raw.csv'
TEST_PATH='lab3/data/sst_test_raw.csv'
VALID_PATH='lab3/data/sst_valid_raw.csv'
VECTORS_PATH='lab3/data/ssr_glove_6b_300d.txt'


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
        
        if usespecialsigns:
           mapping = {string: code for code, string in enumerate(sorted_strings, start=2)}
           spec = {
                '<PAD>': 0,
                '<UNK>': 1
            }
           mapping = {**spec, **mapping}
        else:
            mapping = {string: code for code, string in enumerate(sorted_strings, start=0)}
    
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
        self.instances=[Instance(text.split(), label.strip()) for text, label in data_frame.values] #lista Instance-ova

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
    

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)  
    lengths = torch.tensor([len(text) for text in texts])
    texts = nn.utils.rnn.pad_sequence(sequences=texts, batch_first=True, padding_value=pad_index)  
    return texts, torch.tensor(labels), lengths


def get_embedding_matrix():
    #finish..
    print("dkj")


if __name__=='__main__':
    #extract all words and labels from TRAIN datatset
    df = pd.read_csv(TRAIN_PATH)
    text_corpus = df.iloc[:, :-1].values
    all_words=[]
    for text in text_corpus:
        all_words.extend(text[0].split())
    all_labels = df.iloc[:, -1].values
    all_labels=[la.strip() for la in all_labels]

    #intitialize text vocabulary and labels vocabulary, based on TRAIN data
    text_vocab=Vocab(corpus=all_words)
    labels_vocab=Vocab(corpus=all_labels, usespecialsigns=False)

    #intialize the NLPDatasets for train test and valid
    train_dataset = NLPDataset(csv_file=TRAIN_PATH, text_vocab=text_vocab, label_vocab=labels_vocab)
    test_dataset = NLPDataset(csv_file=TEST_PATH, text_vocab=text_vocab, label_vocab=labels_vocab)
    valid_dataset = NLPDataset(csv_file=VALID_PATH, text_vocab=text_vocab, label_vocab=labels_vocab)

    #create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    texts, labels, lengths = next(iter(train_loader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")


