import time
import torch
import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


# model= resnet34(num_classes=4)
model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 4})
# model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=4, pretrained=None)
print(model)

# setattr(model, 'avg_pool', GeM())
# print(hasattr(model, 'avgpool'))
# avg_pool_layer = model.avgpool
# avg_pool_layer = GeM()
# model.avgpool = GeM()
# print(model)

# print(model._conv_head)
# for name, child in model.named_children():
#     print(name)
    # if name == 'head':
    #     for param in child.parameters():
    #         param.requires_grad = True
    # else:
    #     for param in child.parameters():
    #         param.requires_grad = False

# print('#############################################################################################')
# print(model._fc)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # print(count_parameters(model))
    _input = torch.rand(1, 3, 128, 128)
    # out = model.features(_input)
    # for i in range (100):
    #     out = model.extract_features(_input)
    #     print(out.shape)
