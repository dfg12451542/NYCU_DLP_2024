import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class my_network(nn.Module):
  def __init__(self):
    super().__init__()
    self.pool = nn.MaxPool2d(2, stride=2)
    self.softmax = nn.Softmax(dim=1)

    self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
    self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    
    self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

    self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
    self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)

    self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)

    self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)

    self.fc1 = nn.Linear(18432, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, 1000)
    self.fc4 = nn.Linear(1000, 100)

  def forward(self, input):
    x = F.relu(self.conv1_1(input))
    x = F.relu(self.conv1_2(x))
    x = self.pool(x)

    x = F.relu(self.conv2_1(x))
    x = F.relu(self.conv2_2(x))
    x = self.pool(x)

    x = F.relu(self.conv3_1(x))
    x = F.relu(self.conv3_2(x))
    x = F.relu(self.conv3_3(x))
    x = F.relu(self.conv3_4(x))
    x = self.pool(x)

    x = F.relu(self.conv4_1(x))
    x = F.relu(self.conv4_2(x))
    x = F.relu(self.conv4_3(x))
    x = F.relu(self.conv4_4(x))
    x = self.pool(x)

    x = F.relu(self.conv5_1(x))
    x = F.relu(self.conv5_2(x))
    x = F.relu(self.conv5_3(x))
    x = F.relu(self.conv5_4(x))
    x = self.pool(x)

    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    #x = self.softmax(x)

    return x