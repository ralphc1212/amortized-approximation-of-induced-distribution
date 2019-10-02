#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

## the pytorch example NN for MNIST
## used in EMNIST experiment

class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout2d(x, p=0.5)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(x, p=0.5)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout2d(x, p=0.5)
        x = self.fc2(x)
        return x

class mnist_net_f(nn.Module):
    def __init__(self):
        super(mnist_net_f, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class mnist_net_g(nn.Module):
    def __init__(self):
        super(mnist_net_g, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class mnist_mlp(nn.Module):
    def __init__(self, dropout =False, out_dim = 10):
        super(mnist_mlp, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, 0.4)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = F.dropout(x, 0.3)
        x = self.fc3(x)
        return x

class mnist_mlp_h(nn.Module):
    def __init__(self, out_dim = 10):
        super(mnist_mlp_h, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class mnist_mlp_g(nn.Module):
    def __init__(self):
        super(mnist_mlp_g, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x