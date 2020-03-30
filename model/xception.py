import torch
from torch import nn
from torch.nn import functional as F
from pretrainedmodels.models.xception import Xception

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        return self.logits(x)

def net():
    state_dict = torch.load("./weight/xception-b5690688.pth")
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    pretrained_model = Xception(num_classes=1000)
    pretrained_model.load_state_dict(state_dict)

    model = nn.Sequential(
        *list(pretrained_model.children())[:-1],
        Flatten(),
    )
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 2),
    )
    return model


if __name__ == '__main__':
    model = net().cuda()
    x = torch.rand(1, 3, 299, 299).cuda()
    print(model(x).shape)