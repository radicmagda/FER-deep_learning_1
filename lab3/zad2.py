import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from zad1 import get_data_loaders_and_emb_mat

class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, freeze=True, padding_idx=0):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        self.embedding_dim = embedding_matrix.shape[1]
        
        self.fc1 = nn.Linear(self.embedding_dim, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x, lengths):
       """
       x -> (B, max_seq_length) or (max_seq_length)
       lengths -> (B,) or ()

       """
       if len(x.shape) == 1:                   # u slucaju jednog inputa
            x = x.unsqueeze(0)
            lengths = lengths.unsqueeze(0)

       embedded_x=self.embedding(x)            #Shape: (B, max_seq_length ,embedding_dim)
       sum_embeddings = embedded_x.sum(dim=1)  # Shape: (B, embedding_dim)
       lengths = lengths.unsqueeze(-1).float()  # Shape: (B, 1)
       average_embeddings = sum_embeddings / lengths  # Shape: (B, embedding_dim)
       x=average_embeddings
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = F.relu(x)
       x = self.fc3(x)
       return x.flatten().squeeze()    # shape (B,) of batch, () of single input


def train(model, data, optimizer, criterion, clip):
  model.train()
  for batch_num, batch in enumerate(data):
    x,y,lengths=batch
    model.zero_grad()
    # ...
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    # ...


def evaluate(model, data, criterion, args):
  model.eval()
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      # ...
      logits = model(x)
      loss = criterion(logits, y)
      # ...

def main(seed, epochs, clip):
  np.random.seed(seed)
  torch.manual_seed(seed)

  train_loader, valid_loader, test_loader, em=get_data_loaders_and_emb_mat(b_size_train=10, b_size_valid=32, b_size_test=32, random_init=False)
  model = BaselineModel(input_size=300, embedding_matrix=em)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  for epoch in range(epochs):
    train(model=model, data=train_loader, optimizer=optimizer, criterion=criterion, clip=clip)
    evaluate(...)

if __name__=="__main__":
  main(seed=928, epochs=5, clip=6)