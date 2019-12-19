import torch
import numpy as np


def sensitivity(y_pred, y_gt, target):
    eps = 1e-20

    p = 0
    tp = 0
    for i, pred in enumerate(y_pred):
        if y_gt[i] == target:
            p += 1
            if y_pred[i] == target:
                tp += 1
    sens = tp / (p + eps)
    return sens


def specificity(y_pred, y_gt, target):
    eps = 1e-20

    n = 0
    tn = 0
    for i, pred in enumerate(y_pred):
        if not y_gt[i] == target:
            n += 1
            if not y_pred[i] == target:
                tn += 1
    spec = tn / (n + eps)
    return spec


def accuracy(y_pred, y_gt):
    eps = 1e-20

    pn = 0
    tptn = 0
    for i, pred in enumerate(y_pred):
        pn += 1
        if y_gt[i] == pred:
            tptn += 1
    acc = tptn / (pn + eps)
    return acc


def evaluate(prediction, ground_truth, num_classes=4): ## 이부분만 평가 기준에 맞게 변경할 것

    # performance
    # class_acc = []
    class_sens = []
    class_spec = []

    for i in range(num_classes):
        class_sens.append(sensitivity(prediction, ground_truth, i))
        class_spec.append(specificity(prediction, ground_truth, i))
        # class_acc.append((class_sens[i] + class_spec[i]) / 2)

    ttl_sens = sum(class_sens) / len(class_sens)
    ttl_spec = sum(class_spec) / len(class_spec)
    ttl_acc = accuracy(prediction, ground_truth)

    res_acc = round(ttl_acc, 4) * 100
    res_sens = round(ttl_sens, 4) / 100
    res_spec = round(ttl_spec, 4) / 1000000

    res = round(res_acc + res_sens + res_spec, 10)

    return res, class_sens, class_spec, ttl_acc, ttl_sens, ttl_spec




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import time

    torch.manual_seed(2019)
    np.random.seed(2019)

    logits = torch.rand(16, 4)
    labels = torch.randint(0, 4, [16])
    preds = logits.argmax(dim=1)
    print(preds)
    print(labels)

    # preds = preds.view((-1, 1))
    # labels = labels.view((-1, 1))
    print(evaluate(4, preds, labels))


    # print(topk_accuracy(logits,labels, topk=(1,)))
    # print(calc_scores(logits, labels, reduce=True))

    '''
    preds = logits.argmax(dim=1)

    nb_classes = 4
    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(labels, preds):
        conf_matrix[t, p] += 1

    # print('Confusion matrix\n', conf_matrix)

    TP = conf_matrix.diag()

    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[idx.nonzero()[:,
                         None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()

        # print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        #     c, TP[c], TN, FP, FN))
    '''