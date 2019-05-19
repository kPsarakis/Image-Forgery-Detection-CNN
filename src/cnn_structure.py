import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()

        # # Convolution 2
        # self.conv = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=2, padding=0)
        # self.relu = nn.ReLU()

        # Max pool 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout for regularization 75%
        self.dropout1 = nn.Dropout(p=0.75)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=16, kernel_size=5, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=30, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        # Convolution 4
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Max pool 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout for regularization 75%
        self.dropout2 = nn.Dropout(p=0.75)

        # Convolution 5
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()

        # Convolution 6
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu6 = nn.ReLU()

        # Convolution 7
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        # Convolution 8
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu8 = nn.ReLU()

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=5 * 5 * 16 , out_features=100)

        # Dropout for regularization 50%
        self.dropout3 = nn.Dropout(p=0.5)

        # softmax

    def forward(self, x):

        # Convolution 1
        x = self.conv1(x)
        x = self.relu1(x)

        # # Convolution 2
        # x = self.conv(x)
        # x = self.relu(x)

        # Max pool 1
        x = self.pool1(x)

        # # Dropout for regularization 75%
        # x = self.dropout1(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.relu2(x)

        # Convolution 3
        x = self.conv3(x)
        x = self.relu3(x)

        # Convolution 4
        x = self.conv4(x)
        x = self.relu4(x)

        # Max pool 2
        x = self.pool(x)

        # # Dropout for regularization 75%
        # x = self.dropout2(x)

        # Convolution 5
        x = self.conv5(x)
        x = self.relu5(x)

        # Convolution 6
        x = self.conv6(x)
        x = self.relu6(x)

        # Convolution 7
        x = self.conv7(x)
        x = self.relu7(x)

        # Convolution 8
        x = self.conv8(x)
        x = self.relu8(x)


        # Fully Connected Layer
        x = self.fc(x)

        # Dropout for regularization 50%
        x = self.dropout3(x)

        # softmax

        return x


net = Net()

# print(net.conv1.weight.shape)
