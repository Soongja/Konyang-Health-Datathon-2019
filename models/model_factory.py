import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50
# from .resnet9 import resnet9
from .resnet_se import se_resnet18, se_resnet34
from .cadene.senet import se_resnet50, se_resnext50_32x4d, se_resnext101_32x4d
from .cadene.torchvision_models import densenet121
from .efficientnet.model import EfficientNet
from .gem import GeM


def get_model(config, num_classes=4):
    model_name = config.MODEL.NAME

    if model_name.startswith('resnet'):
        model = globals().get(model_name)(num_classes=num_classes)
    elif model_name.startswith('efficient'):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
    elif model_name.startswith('se_resnet'):
        model = globals().get(model_name)(num_classes=num_classes)
    else:
        model = globals().get(model_name)(num_classes=num_classes, pretrained=None)

    # average pooling
    avgpool = 'avgpool' if hasattr(model, 'avgpool') else 'avg_pool'
    setattr(model, avgpool, GeM()) if config.MODEL.GEM else setattr(model, avgpool, nn.AdaptiveAvgPool2d(1))

    # fc
    if hasattr(model, 'fc'):
        fc = 'fc'
        in_features = model.fc.in_features
    elif hasattr(model, '_fc'):
        fc = '_fc'
        in_features = model._fc.in_features
    elif hasattr(model, 'last_linear'):
        fc = 'last_linear'
        in_features = model.last_linear.in_features

    double_fc  = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True,
                                   track_running_stats=True),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes))

    setattr(model, fc, double_fc) if config.MODEL.DOUBLE_FC else setattr(model, fc, nn.Linear(in_features, num_classes))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # if config.PARALLEL:
    #     model = nn.DataParallel(model)

    print('model name:', model_name)
    return model
