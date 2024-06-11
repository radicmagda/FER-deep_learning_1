import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn.functional as F
from zad1 import get_data_loaders_and_emb_mat

REUSULTS_PATH='lab3/results_zad3.txt'

class Args:
    def __init__(self, seed, epochs, batch_size, lr, clip, log_interval):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip
        self.log_interval = log_interval

class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, freeze=True, padding_idx=0):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        self.embedding_dim = embedding_matrix.shape[1]
        self.lstm1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=150, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=150, hidden_size=150, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(150, 150)
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x, lengths):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            lengths = lengths.unsqueeze(0)
        
        embedded_x = self.embedding(x)  # Shape: (B, max_seq_length, embedding_dim)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded_x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm1(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm2(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        hn = hn[-1]  # use the hidden state from the last layer of LSTM
        x = self.fc1(hn)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.flatten().squeeze()
        return x

def train(model, data, optimizer, criterion, args):
    model.train()
    total_loss = 0
    for batch_num, (data_batch, labels_batch, lengths_batch) in enumerate(data):
        optimizer.zero_grad()
        logits = model(data_batch, lengths_batch)
        loss = criterion(logits, labels_batch.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch_num % args.log_interval == 0:
            print(f'Batch {batch_num}, Loss: {loss.item()}')
    return total_loss / len(data)

def evaluate(model, data, criterion, args):
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch_num, (data_batch, labels_batch, lengths_batch) in enumerate(data):
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, valid_loader, test_loader, em=get_data_loaders_and_emb_mat(b_size_train=10, 
                                                                           b_size_valid=32, 
                                                                           b_size_test=32, 
                                                                           random_init=False)
    model = LSTMModel(embedding_matrix=em, freeze=True)  #freeze=not random_init

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    with open(REUSULTS_PATH, 'a') as f:
        f.write(f'Run {run + 1} with Seed {args.seed}\n')
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, criterion, args)
            valid_loss, valid_accuracy, valid_f1, valid_cm = evaluate(model, valid_loader, criterion, args)
            f.write(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}, Valid F1: {valid_f1}\n')
            f.write(f'Confusion Matrix:\n{valid_cm}\n\n')

    test_loss, test_accuracy, test_f1, test_cm = evaluate(model, test_loader, criterion, args)
    return test_accuracy, test_f1, test_cm

if __name__ == "__main__":
    seeds = [42, 43, 44, 45, 46]
    results = []
    for run, seed in enumerate(seeds):
        args = Args(seed=seed, epochs=5, batch_size=32, lr=1e-4, clip=0.25, log_interval=100)
        accuracy, f1, cm = main(args, run)
        results.append((args.seed, accuracy, f1, cm))

    with open(REUSULTS_PATH, 'a') as f:
        f.write('Final Test Results:\n')
        for seed, accuracy, f1, cm in results:
            f.write(f'Seed: {seed}, Accuracy: {accuracy}, F1: {f1}, Confusion Matrix: \n{cm}\n\n')
        f.write(f'Arguments: {vars(args)}\n')
