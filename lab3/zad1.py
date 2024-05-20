import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class NLPDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Assuming the last column is the target and the rest are features
        self.features = self.data_frame.iloc[:, :-1].values
        self.labels = self.data_frame.iloc[:, -1].values
        
        # Standardize the features (optional)
        #.. finish implementation

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = {'features': torch.tensor(self.features[idx], dtype=torch.float32),
                  'labels': torch.tensor(self.labels[idx], dtype=torch.float32)}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Instantiate the datasets
train_dataset = NLPDataset('data/sst_train_raw.csv')
valid_dataset = NLPDataset('data/sst_valid_raw.csv')
test_dataset = NLPDataset('data/sst_test_raw.csv')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example of iterating through the data loaders
for batch in train_loader:
    features, labels = batch['features'], batch['labels']
    print(features, labels)
    # Training loop logic here
