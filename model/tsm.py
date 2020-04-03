import torch
from model.resnet import resnet50
from model.basic_ops import SegmentConsensus
from torch import nn

class DivSegment(nn.Module):
    def __init__(self, num_classes=2):
        super(DivSegment, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x.view(-1, 5, self.num_classes)

class TSM(nn.Module):

    def __init__(self, num_classes=2):
        super(TSM, self).__init__()


        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )
        self.divsegment = DivSegment(num_classes=num_classes)
        self.consensus = SegmentConsensus(consensus_type='avg')


    def forward(self, x):
        x = self.model(x)
        x = self.divsegment(x)
        x = self.consensus(x)
        return x.squeeze(1)

if __name__ == '__main__':
    model = TSM(num_classes=2)

    x = torch.rand(3*5, 3, 224, 224)
    print(model(x).shape)