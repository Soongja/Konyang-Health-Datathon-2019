import os
import sys
import numpy as np
import torch


def prepare_train_directories(config):
    out_dir = config.LOCAL_TRAIN_DIR
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    # os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageMeterArray(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.val = np.zeros(self.num_classes)
        self.avg = np.zeros(self.num_classes)
        self.sum = np.zeros(self.num_classes)
        self.count = 0

    def update(self, val, n=1):
        assert self.val.shape == val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else np.zeros(self.num_classes)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
