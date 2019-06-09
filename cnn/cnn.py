import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, img_size=56, num_classes=36):
        super(CNN, self).__init__()

        self.img_size = img_size
        linear_size = (16 * (((self.img_size - 4) / 2 - 4) / 2) * (((self.img_size - 4) / 2 - 4) / 2))
        linear_size = int(linear_size)
        self.linear_size = linear_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(linear_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.img_size, self.img_size)

        # input (1, img_size, img_size) output (8, img_size - 4, img_size - 4)
        x = F.relu(self.conv1(x))

        # input (8, img_size - 4, img_size -4) output (8, (img_size - 4)/2, (img_size - 4)/2)
        x = self.maxpool(x)

        # input (8, (img_size - 4)/2, (img_size - 4)/2) output (16, (img_size - 4)/2 - 4, (img_size - 4)/2 - 4)
        x = F.relu(self.conv2(x))

        # input (16, (img_size - 4)/2 - 4, (img_size - 4)/2 - 4)
        # output (16, ((img_size - 4)/2 - 4)/2, ((img_size - 4)/2 - 4)/2)
        x = self.maxpool(x)

        x = x.view(-1, self.linear_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
