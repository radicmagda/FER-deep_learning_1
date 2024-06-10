import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from zad1 import get_data_loaders_and_emb_mat
from collections import namedtuple

Args = namedtuple('Args', ['seed', 'epochs', 'batch_size', 'lr', 'clip', 'log_interval'])

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


def train(model, data, optimizer, criterion, args):
    model.train()
    total_loss = 0
    for batch_num, batch in enumerate(data):
        data_batch, labels_batch, lengths_batch = batch
        labels_batch = labels_batch.float()  # convert labels to float, so its compattoble with logits

        model.zero_grad()
        logits = model(data_batch, lengths_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        if batch_num % args.log_interval == 0 and batch_num > 0:
            print(f'Batch {batch_num}/{len(data)}, Loss: {loss.item()}')

    return total_loss / len(data)


def evaluate(model, data, criterion, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            data_batch, labels_batch, lengths_batch = batch
            labels_batch = labels_batch.float()  # convert labels to float so its compattible with logits
            logits = model(data_batch, lengths_batch)
            loss = criterion(logits, labels_batch)

            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits))
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

    accuracy = correct / total
    return total_loss / len(data), accuracy

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  train_loader, valid_loader, test_loader, em=get_data_loaders_and_emb_mat(b_size_train=10, 
                                                                           b_size_valid=32, 
                                                                           b_size_test=32, 
                                                                           random_init=False)
  model = BaselineModel(embedding_matrix=em, freeze=True)  #freeze=not random_init

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


  # Training loop
  for epoch in range(args.epochs):
      train_loss = train(model, train_loader, optimizer, criterion, args)
      valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, args)
      print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}')

    # Test evaluation
  test_loss, test_accuracy = evaluate(model, test_loader, criterion, args)
  print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

if __name__=="__main__":
    args = Args(seed=42, epochs=5, batch_size=32, lr=1e-4, clip=1.0, log_interval=100)
    main(args)
    