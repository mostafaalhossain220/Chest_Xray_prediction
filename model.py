import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.batch1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3)
        self.batch4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 512, 3)
        self.batch5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(5 * 5 * 512, 1024)
        self.fc1_batch = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 256)
        self.fc2_batch = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.fc3_batch = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.pool1(F.relu(self.batch1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch2(self.conv2(x))))
        x = self.pool3(F.relu(self.batch3(self.conv3(x))))
        x = self.pool4(F.relu(self.batch4(self.conv4(x))))
        x = self.pool5(F.relu(self.batch5(self.conv5(x))))

        x = self.flatten(x)

        x = F.relu(self.fc1_batch(self.fc1(x)))
        x = F.relu(self.fc2_batch(self.fc2(x)))
        x = F.relu(self.fc3_batch(self.fc3(x)))

        x = self.fc4(x)

        return x