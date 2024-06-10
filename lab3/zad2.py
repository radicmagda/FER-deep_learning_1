import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from zad1 import get_data_loaders_and_emb_mat
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

class Args:
    def __init__(self, seed, epochs, batch_size, lr, clip, log_interval):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip
        self.log_interval = log_interval

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
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for data_batch, labels_batch, lengths_batch in data:
            labels_batch=labels_batch.float()
            logits = model(data_batch, lengths_batch)
            loss = criterion(logits, labels_batch.float())
            total_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(labels_batch)
    avg_loss = total_loss / len(data)
    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu()
    predictions = torch.round(torch.sigmoid(all_logits))
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    cm = confusion_matrix(all_labels, predictions)
    return avg_loss, accuracy, f1, cm

def main(args, run):
  print(f'\n------------Run: {run+1}')
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  train_loader, valid_loader, test_loader, em=get_data_loaders_and_emb_mat(b_size_train=10, 
                                                                           b_size_valid=32, 
                                                                           b_size_test=32, 
                                                                           random_init=False)
  model = BaselineModel(embedding_matrix=em, freeze=True)  #freeze=not random_init

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  with open('results.txt', 'a') as f:
        f.write(f'Run {run + 1} with Seed {args.seed}\n')
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, criterion, args)
            valid_loss, valid_accuracy, valid_f1, valid_cm = evaluate(model, valid_loader, criterion, args)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}')
            f.write(f'Epoch {epoch + 1}') 
            f.write(f'Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}, Valid F1: {valid_f1}\n')
            f.write(f'ValidConfusion Matrix:\n{valid_cm}\n\n')
    
    # final evaluation on test set
  test_loss, test_accuracy, test_f1, test_cm = evaluate(model, test_loader, criterion, args)
  print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
  return test_accuracy, test_f1, test_cm

if __name__=="__main__":
    results = []
    for run in range(5):
        args = Args(seed=7052020+run, epochs=5, batch_size=32, lr=1e-4, clip=1.0, log_interval=100)
        accuracy, f1, cm = main(args, run)
        results.append((args.seed, accuracy, f1, cm))

    # Save final test results to file
    with open('results.txt', 'a') as f:
        f.write('Final Test Results:\n')
        for seed, accuracy, f1, cm in results:
            f.write(f'Seed: {seed}, Accuracy: {accuracy}, F1: {f1}, Confusion Matrix: \n{cm}\n\n')
        f.write(f'Arguments: {vars(args)}\n')
    