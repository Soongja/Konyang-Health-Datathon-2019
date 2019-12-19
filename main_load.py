import os
import argparse
import pprint
import sys
import time
import random
import cv2
import numpy as np
import torch
import warnings

import utils.config
from models.model_factory import get_model
from datasets.dataloader import get_dataloader
from factory.losses import get_loss
from factory.schedulers import get_scheduler
from factory.optimizers import get_optimizer
from factory.transforms import Albu
from core.calculate_mean_std import train_std
from core.preprocess import preprocess
from core.split_cv import split_cv
from core.train import train_single_epoch, evaluate_single_epoch
from core.test import _infer
from utils.experiments import group_weight

import nsml
try:
    from nsml.constants import DATASET_PATH, GPU_NUM
    IS_LOCAL = False
except:
    import utils.checkpoint
    from utils.tools import prepare_train_directories
    DATASET_PATH = 'data'
    IS_LOCAL = True

warnings.filterwarnings("ignore")


yml = 'configs/base.yml'
config = utils.config.load(yml)


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model.pth')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode  ## 해당 부분은 data loader의 infer_func을 의미
        # 여기 들어오는 data는 정해져 있는데 list이고 cv2.imread(f, 3)한 np array들이 담겨 있어.
        # baseline 코드 새로 나오면 바뀔 수 있음.
        return _infer(model, data, config)

    nsml.bind(save=save, load=load, infer=infer)


def get_args():
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    return config


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything()

    pprint.pprint(config, indent=2)

    model = get_model(config).cuda()
    bind_model(model)

    args = get_args()
    if args.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if args.mode == 'train':  ### training mode일 때
        print('Training Start...')

        nsml.load(checkpoint='18', session='team146/KHD2019_FUNDUS/20')
        nsml.save(0)
        exit()


if __name__ == '__main__':
    main()
