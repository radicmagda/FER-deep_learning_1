import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

class Convolution(nn.Module):
    def __init__(self, num_filters, kernel_size, padding='SAME'):
        super(Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class FC(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FC, self).__init__()
        self.fc = nn.Linear(num_inputs, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Convolution(num_filters=16, kernel_size=5)
        self.conv2 = Convolution(num_filters=32, kernel_size=5)
        self.fc3 = FC(num_inputs=32*4*4, num_outputs=512)
        self.logits = FC(num_inputs=512, num_outputs=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc3(x)
        x = self.logits(x)
        return x

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(valid_loader, model, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(valid_loader)
    return accuracy, avg_loss

def main():
    np.random.seed(int(time.time() * 1e6) % 2**31)

    transform = ToTensor()
    train_dataset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(8):
        train(train_loader, model, criterion, optimizer, device)
        train_accuracy, train_loss = evaluate(train_loader, model, criterion, device)
        valid_accuracy, valid_loss = evaluate(valid_loader, model, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

    test_accuracy, _ = evaluate(test_loader, model, criterion, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
