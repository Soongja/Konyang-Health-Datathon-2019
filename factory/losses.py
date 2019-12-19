import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def CELoss():
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, num_classes=4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, labels):
        log_preds = F.log_softmax(logits, dim=1)
        preds = log_preds.exp()

        labels = F.one_hot(labels, self.num_classes).float().cuda()

        L = (-labels) * (1 - preds)**self.gamma * log_preds
        L = L.sum(dim=1)
        L = L.mean()

        return L


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=4):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, labels):
        pred = self.softmax(pred)
        label_one_hot = F.one_hot(labels, self.num_classes).float().cuda()

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        ce = (-1*torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        loss = self.alpha * ce + self.beta * rce

        return loss


########################################################################################################################


def get_loss(loss_name):
    print('loss name:', loss_name)
    f = globals().get(loss_name)
    return f()


if __name__ == '__main__':
    import time

    seed = 2019
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    input = torch.rand(16, 4).cuda()
    target = torch.randint(0, 4, [16]).cuda()

    loss = FocalLoss()
    # loss = SCELoss(alpha=0.1, beta=0.1)
    print(loss(input, target))

    # start = time.time()
    # for i in range(100):
        # out = binary_lovasz_loss()(input, target)
        # out = binary_lovasz()(input, target)
        # out = binary_lovasz2()(input, target)
    # print(time.time() - start)
