import torch
from torch import nn


class MesoInception(nn.Module):

    def __init__(self, out_channel=2):
        super(MesoInception, self).__init__()

        self.inception1_1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.inception1_2 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.inception1_3 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
        )
        self.inception1_4 = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
        )
        self.inception1_bn = nn.BatchNorm2d(11)
        self.inception1_maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.inception2_1 = nn.Conv2d(11, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.inception2_2 = nn.Sequential(
            nn.Conv2d(11, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.inception2_3 = nn.Sequential(
            nn.Conv2d(11, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
        )
        self.inception2_4 = nn.Sequential(
            nn.Conv2d(11, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
        )
        self.inception2_bn = nn.BatchNorm2d(12)
        self.inceoptin2_maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.block3 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, stride=1, padding=2, bias=False),
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
        self.leakrelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16, out_channel),
        )

    def InceptionLayer1(self, input):
        x1 = self.inception1_1(input)
        x2 = self.inception1_2(input)
        x3 = self.inception1_3(input)
        x4 = self.inception1_4(input)

        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.inception1_bn(out)
        out = self.inception1_maxpool(out)

        return out

    def InceptionLayer2(self, input):
        x1 = self.inception2_1(input)
        x2 = self.inception2_2(input)
        x3 = self.inception2_3(input)
        x4 = self.inception2_4(input)

        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.inception2_bn(out)
        out = self.inceoptin2_maxpool(out)

        return out

    def forward(self, input):
        x = self.InceptionLayer1(input)
        x = self.InceptionLayer2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leakrelu(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    net = MesoInception(out_channel=2).cuda()
    x = torch.rand(2, 3, 256, 256).cuda()
    print(net(x).shape)