import torch


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
        torch.nn.Conv2d(channels_in, channels_out,
                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def build_network(num_classes=10):
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resnet9(num_classes=4):
    return build_network(num_classes)


if __name__ == '__main__':
    net = build_network(4)
    print(net)
    print(count_parameters(net))
    print(resnet9(4))