"""
Reference:
- https://github.com/khornlund/severstal-steel-defect-detection/blob/master/sever/model/loss.py
"""

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, Conv2d, Linear


class LabelSmoother:
    """
    Maps binary labels (0, 1) to (eps, 1 - eps)
    """
    def __init__(self, eps=1e-4):
        self.eps = eps
        self.scale = 1 - 2 * self.eps
        self.bias = self.eps / self.scale

    def __call__(self, t):
        return (t + self.bias) * self.scale


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay
