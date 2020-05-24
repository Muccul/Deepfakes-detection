import torch
from model.resnet import resnet50
from model.basic_ops import SegmentConsensus
from torch import nn

class DivSegment(nn.Module):
    def __init__(self, num_classes=2, n_segment=5):
        super(DivSegment, self).__init__()
        self.num_classes = num_classes
        self.n_segment = n_segment

    def forward(self, x):
        return x.view(-1, self.n_segment, self.num_classes)

class TSM(nn.Module):

    def __init__(self, input_channel=3, num_classes=2, n_segment=5, pretrained=False):
        super(TSM, self).__init__()


        self.model = resnet50(pretrained=pretrained, n_segment=n_segment, input_channel=input_channel)
        # make_non_local(self.model, n_segment=5)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )
        self.divsegment = DivSegment(num_classes=num_classes, n_segment=n_segment)
        self.consensus = SegmentConsensus(consensus_type='avg')


    def forward(self, x, div=False):
        x = self.model(x)
        x_div = self.divsegment(x)
        x = self.consensus(x_div)

        if div == True:
            return x.squeeze(1), x_div
        return x.squeeze(1)

if __name__ == '__main__':

    model = TSM(num_classes=2, n_segment=4, input_channel=2, pretrained=False)
    x = torch.rand(4, 2, 224, 224)
    print(model(x).shape)

