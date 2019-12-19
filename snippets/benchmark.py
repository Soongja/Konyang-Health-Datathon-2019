import time
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from efficientnet_pytorch import EfficientNet
import pretrainedmodels


def speed(model, name):
    # error 방지
    ###################################################
    if name.startswith('resnet'):
        model.avgpool = nn.AdaptiveAvgPool2d(1)

    elif name.startswith('efficient'):
        pass

    else:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    ###################################################

    model.cuda()
    input = torch.rand(1, 3, 256, 256).cuda()

    model(input)
    t2 = time.time()

    for i in range(10):
        model(input)
    t3 = time.time()

    print('%s : %f' % (name, t3 - t2))


# pretrainedmodels
# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2', 'alexnet',
# 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
# 'vgg19_bn', 'vgg19', 'nasnetalarge', 'nasnetamobile', 'cafferesnet101', 'senet154',  'se_resnet50', 'se_resnet101',
# 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'polynet', 'pnasnet5large']

if __name__ == '__main__':
    resnet18 = resnet18()
    resnet34 = resnet34()
    resnet50 = resnet50()
    effb0 = EfficientNet.from_name('efficientnet-b0')
    effb1 = EfficientNet.from_name('efficientnet-b1')
    effb2 = EfficientNet.from_name('efficientnet-b2')
    effb3 = EfficientNet.from_name('efficientnet-b3')
    effb4 = EfficientNet.from_name('efficientnet-b4')
    effb5 = EfficientNet.from_name('efficientnet-b5')
    effb6 = EfficientNet.from_name('efficientnet-b6')
    effb7 = EfficientNet.from_name('efficientnet-b7')
    seresnext50 = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
    seresnext101 = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
    inceptionv4 = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained=None)

    speed(resnet18, 'resnet18')
    speed(resnet34, 'resnet34')
    speed(resnet50, 'resnet50')
    speed(effb0, 'efficientnet-b0')
    speed(effb1, 'efficientnet-b1')
    speed(effb2, 'efficientnet-b2')
    speed(effb3, 'efficientnet-b3')
    speed(effb4, 'efficientnet-b4')
    speed(effb5, 'efficientnet-b5')
    speed(effb6, 'efficientnet-b6')
    speed(effb7, 'efficientnet-b7')
    speed(seresnext50, 'se_resnext50_32x4d')
    speed(seresnext101, 'se_resnext101_32x4d')
    speed(inceptionv4, 'inceptionv4')
