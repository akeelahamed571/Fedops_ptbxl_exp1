import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRNNClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_classes=71):
        super(ConvRNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        return self.fc(h_n)

def train_torch():
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return train

def test_torch():
    def test(model, test_loader, criterion, device):
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == torch.argmax(targets, dim=1)).sum().item()
                total += targets.size(0)
        return correct / total
    return test
