import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=5) #500x500x3 to 100x100x6
        self.pool = nn.MaxPool2d(2, stride=2) #100x100x1 to 50x50x1
        self.conv2 = nn.Conv2d(6, 12, 2, stride=2) #50x50x1 to 25x25x1
        self.fc1 = nn.Linear(25*25*12, 3750)
        self.fc2 = nn.Linear(3750, 1875)
        self.fc3 = nn.Linear(1875, 375)
        self.fc4 = nn.Linear(375, 75)
        self.fc5 = nn.Linear(75, 15)
        self.fc6 = nn.Linear(15, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 25*25*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x
