import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import os
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from pathlib import Path

def save_conv1_kernels(epoch, model, save_dir):
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    num_filters, in_channels, kernel_size, _ = conv1_weights.shape
    k = kernel_size
    w = conv1_weights.reshape(num_filters, in_channels, k, k)

    w -= w.min()
    w /= w.max()

    border = 1
    cols = 8
    rows = int(np.ceil(num_filters / cols))

    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border

    img = np.zeros([height, width])

    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k] = w[i, 0]  # Assuming input has only 1 channel (grayscale)

    filename = f'conv1_epoch_{epoch}.png'
    img_uint8 = TF.to_pil_image(img)
    img_uint8.save(os.path.join(save_dir, filename))


class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, pool1_size, conv2_width, pool2_size, fc1_width, class_count):
        super(ConvolutionalModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(pool1_size, stride=pool1_size)

        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(pool2_size, stride=pool2_size)

        self.fc1_width = fc1_width
        self.fc1 = nn.Linear(conv2_width * (28 // pool1_size // pool2_size)**2, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool2(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = torch.relu(h)

        logits = self.fc_logits(h)
        return logits


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
SAVE_DIR=Path(__file__).parent / 'out_torch_model'
batch_size = 50
learning_rate = 1e-3
weight_decay=1e-3
num_epochs = 8

train_dataset = MNIST(root='.', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='.', train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = ConvolutionalModel(in_channels=1, conv1_width=16, pool1_size=2, conv2_width=32, pool2_size=2, fc1_width=512, class_count=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

start_time = time.time()
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
