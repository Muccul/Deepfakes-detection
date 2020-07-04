import torch
from model.resnet import resnet50
import torchvision
from model.non_local import make_non_local

if __name__ == '__main__':
    model = resnet50(pretrained=True)
    make_non_local(model, n_segment=5)

    x = torch.rand(1*5, 3, 224, 224)
    print(model(x).shape)