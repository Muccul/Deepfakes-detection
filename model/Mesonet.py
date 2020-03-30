import torch
from torch import nn
from torch.nn import functional as F


class MesoNet(nn.Module):

    def __init__(self, out_channel=2):
        super(MesoNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 16),
        )
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16, out_channel),
        )

    def Flatten(self, input):
        return input.view(input.size(0), -1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    net = MesoNet(out_channel=2).cuda()
    x = torch.rand(2, 3, 256, 256).cuda()
    print(net(x).shape)