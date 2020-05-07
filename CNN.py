import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 24, 5)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.pool3 = nn.MaxPool2d(2, stride=1)
        self.lin_layer_input_size = 85*85*24
        self.fc1 = nn.Linear(self.lin_layer_input_size, 416)
        self.fc2 = nn.Linear(416, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, self.lin_layer_input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
