import torch
from model.resnet import resnet50
from model.basic_ops import SegmentConsensus
from torch import nn
from model.non_local import make_non_local

class DivSegment(nn.Module):
    def __init__(self, num_classes=2, n_segment=5):
        super(DivSegment, self).__init__()
        self.num_classes = num_classes
        self.n_segment = n_segment

    def forward(self, x):
        return x.view(-1, self.n_segment, self.num_classes)

class TSM(nn.Module):

    def __init__(self, num_classes=2, n_segment=5):
        super(TSM, self).__init__()


        self.model = resnet50(pretrained=True, n_segment=n_segment)
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

    import os



    device = torch.device("cuda:0")
    weight_path = "weight"
    model = TSM(num_classes=2, n_segment=5)

    # print(model)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)

    model.load_state_dict(
        torch.load(os.path.join('../', weight_path, "tsm_model_Face2Face_c23_C_10.pth"), map_location=device))
    x = torch.rand(5*2, 3, 224, 224).to(device)
    print(model(x).shape)

