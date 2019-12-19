import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.metrics import evaluate
from utils.tools import AverageMeter, AverageMeterArray
from utils.experiments import LabelSmoother


def evaluate_single_epoch(config, model, dataloader, criterion, epoch):
    batch_time = AverageMeter()

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)

            labels = labels.squeeze(1)

            all_logits.append(logits)
            all_labels.append(labels)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_EVERY == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), batch_time.sum))

            del images, labels, logits
            torch.cuda.empty_cache()

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        loss = criterion(all_logits, all_labels)

        all_preds = all_logits.argmax(dim=1)
        all_preds = all_preds.detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()
        res, class_sens, class_spec, ttl_acc, ttl_sens, ttl_spec = evaluate(all_preds, all_labels)
        print(
            ' | %12s | %.4f |\n' % ('loss', loss.item()),
            '| %12s | %.10f |\n' % ('res', res),
            # '| %12s | %.4f %.4f %.4f %.4f |\n' % ('class_acc', class_acc[0], class_acc[1], class_acc[2], class_acc[3]),
            '| %12s | %.4f %.4f %.4f %.4f |\n' % ('class_sens', class_sens[0], class_sens[1], class_sens[2], class_sens[3]),
            '| %12s | %.4f %.4f %.4f %.4f |\n' % ('class_spec', class_spec[0], class_spec[1], class_spec[2], class_spec[3]),
            '| %12s | %.4f |\n' % ('acc', ttl_acc),
            '| %12s | %.4f |\n' % ('sens', ttl_sens),
            '| %12s | %.4f |\n' % ('spec', ttl_spec),
        )

        nb_classes = 4
        conf_matrix = np.zeros((nb_classes, nb_classes))
        for t, p in zip(all_labels, all_preds):
            conf_matrix[t, p] += 1
        # 세로축이 정답지, 가로축이 예측
        print('Confusion Matrix')
        print(conf_matrix)
        print()

    return loss.item(), res, ttl_acc, ttl_sens, ttl_spec


def train_single_epoch(config, model, dataloader, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    reses = AverageMeter()
    accs = AverageMeter()
    senses = AverageMeter()
    specs = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.cuda()
        labels = labels.cuda()
        n_data = images.shape[0]

        logits = model(images)

        labels = labels.squeeze(1)

        if config.LOSS.LABEL_SMOOTHING:
            smoother = LabelSmoother()
            loss = criterion(logits, smoother(labels))
        else:
            loss = criterion(logits, labels)

        losses.update(loss.item(), n_data)

        loss.backward()
        optimizer.step()

        if config.SCHEDULER.NAME == 'one_cycle_lr':
            scheduler.step()
        preds = logits.argmax(dim=1)
        res, _, _, ttl_acc, ttl_sens, ttl_spec = evaluate(preds, labels)

        reses.update(res, n_data)
        accs.update(ttl_acc, n_data)
        senses.update(ttl_sens, n_data)
        specs.update(ttl_spec, n_data)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_EVERY == 0:
            print('[%d/%d][%d/%d] time: %.2f, loss: %.4f, res: %.4f, acc: %.4f, sens: %.4f, spec: %.4f  [lr: %.6f]'
                  % (epoch, config.TRAIN.NUM_EPOCHS, i, len(dataloader), batch_time.sum,
                     loss.item(), res, ttl_acc, ttl_sens, ttl_spec, optimizer.param_groups[0]['lr']),
                     )

        del images, labels, logits
        torch.cuda.empty_cache()

    return (losses.avg, reses.avg, accs.avg, senses.avg, specs.avg)
