import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    text_lengths = [len(text) for text in texts]
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return padded_texts, torch.stack(labels), torch.tensor(text_lengths)

class SimpleNNModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 150)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        x = x.mean(dim=1)  # Mean pooling over the time dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for batch_num, (texts, labels, text_lengths) in enumerate(data_loader):
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

def evaluate(model, data_loader, criterion):
    model.eval()
    losses, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch_num, (texts, labels, text_lengths) in enumerate(data_loader):
            logits = model(texts)
            loss = criterion(logits.squeeze(), labels.float())
            losses.append(loss.item())
            preds = torch.round(torch.sigmoid(logits.squeeze()))
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.mean(losses), all_preds, all_labels

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Pretpostavka: Implementacija load_dataset funkcije
    train_dataset, valid_dataset, test_dataset = load_dataset()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_size = train_loader.dataset[0][0].shape[1]
    model = SimpleNNModel(input_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        valid_loss, valid_preds, valid_labels = evaluate(model, valid_loader, criterion)
        valid_accuracy = accuracy_score(valid_labels, valid_preds)
        valid_f1 = f1_score(valid_labels, valid_preds)
        valid_cm = confusion_matrix(valid_labels, valid_preds)

        print(f"Epoch {epoch + 1}:")
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        print(f"Validation F1 Score: {valid_f1:.4f}")
        print(f"Validation Confusion Matrix:\n{valid_cm}")

    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_cm = confusion_matrix(test_labels, test_preds)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Confusion Matrix:\n{test_cm}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a simple neural network with mean pooling.")
    parser.add_argument('--seed', type=int, default=7052020, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    args = parser.parse_args()
    main(args)
