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
from factory.transforms import Albu, VFlip, HFlip
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

with open(yml, 'r', encoding='UTF8') as f:
    lines = f.readlines()
    for line in lines:
        print(line, end='')


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

    # yml = 'configs/base.yml'
    # config = utils.config.load(yml)
    # pprint.pprint(config, indent=2)

    model = get_model(config).cuda()
    bind_model(model)

    args = get_args()
    if args.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if args.mode == 'train':  ### training mode일 때
        print('Training Start...')

        # no bias decay
        if config.OPTIMIZER.NO_BIAS_DECAY:
            group_decay, group_no_decay = group_weight(model)
            params = [{'params': group_decay},
                      {'params': group_no_decay, 'weight_decay': 0.0}]
        else:
            params = model.parameters()

        optimizer = get_optimizer(config, params)
        optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR
        if config.OPTIMIZER.NO_BIAS_DECAY:
            optimizer.param_groups[1]['initial_lr'] = config.OPTIMIZER.LR

        ###############################################################################################

        if IS_LOCAL:
            prepare_train_directories(config)
            utils.config.save_config(yml, config.LOCAL_TRAIN_DIR)

            checkpoint = utils.checkpoint.get_initial_checkpoint(config)
            if checkpoint is not None:
                last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
            else:
                print('[*] no checkpoint found')
                last_epoch, score, loss = -1, -1, float('inf')
            print('last epoch:{} score:{:.4f} loss:{:.4f}'.format(last_epoch, score, loss))

        else:
            last_epoch = -1
        ###############################################################################################

        scheduler = get_scheduler(config, optimizer, last_epoch=last_epoch)

        ###############################################################################################
        if IS_LOCAL:
            if config.SCHEDULER.NAME == 'multi_step':
                if config.SCHEDULER.WARMUP:
                    scheduler_dict = scheduler.state_dict()['after_scheduler'].state_dict()
                else:
                    scheduler_dict = scheduler.state_dict()

                milestones = scheduler_dict['milestones']
                step_count = len([i for i in milestones if i < last_epoch])
                optimizer.param_groups[0]['lr'] *= scheduler_dict['gamma'] ** step_count
                if config.OPTIMIZER.NO_BIAS_DECAY:
                    optimizer.param_groups[1]['initial_lr'] *= scheduler_dict['gamma'] ** step_count

            if last_epoch != -1:
                scheduler.step()
        ###############################################################################################
        # for dirname, _, filenames in os.walk(DATASET_PATH):
        #     for filename in filenames:
        #         print(os.path.join(dirname, filename))

        # if preprocessing possible
        preprocess_type = config.DATA.PREPROCESS
        cv2_size = (config.DATA.IMG_W, config.DATA.IMG_H)
        if not IS_LOCAL:
            preprocess(os.path.join(DATASET_PATH, 'train', 'train_data', 'NOR'), os.path.join(preprocess_type, 'NOR'), preprocess_type, cv2_size)
            preprocess(os.path.join(DATASET_PATH, 'train', 'train_data', 'AMD'), os.path.join(preprocess_type, 'AMD'), preprocess_type, cv2_size)
            preprocess(os.path.join(DATASET_PATH, 'train', 'train_data', 'RVO'), os.path.join(preprocess_type, 'RVO'), preprocess_type, cv2_size)
            preprocess(os.path.join(DATASET_PATH, 'train', 'train_data', 'DMR'), os.path.join(preprocess_type, 'DMR'), preprocess_type, cv2_size)
            data_dir = preprocess_type
            # data_dir = os.path.join(DATASET_PATH, 'train/train_data')
        else:  # IS_LOCAL
            data_dir = os.path.join(DATASET_PATH, preprocess_type)

        # eda
        # train_std(data_dir, preprocess_type, cv2_size)

        fold_df = split_cv(data_dir, n_splits=config.NUM_FOLDS)
        val_fold_idx = config.IDX_FOLD

        ###############################################################################################

        train_loader = get_dataloader(config, data_dir, fold_df, val_fold_idx, 'train', transform=Albu())
        val_loader = get_dataloader(config, data_dir, fold_df, val_fold_idx, 'val')

        postfix = dict()
        num_epochs = config.TRAIN.NUM_EPOCHS

        val_acc_list = []
        for epoch in range(last_epoch+1, num_epochs):

            if epoch >= config.LOSS.FINETUNE_EPOCH:
                criterion = get_loss(config.LOSS.FINETUNE_LOSS)
            else:
                criterion = get_loss(config.LOSS.NAME)

            train_values = train_single_epoch(config, model, train_loader, criterion, optimizer, scheduler, epoch)
            val_values = evaluate_single_epoch(config, model, val_loader, criterion, epoch)
            val_acc_list.append((epoch, val_values[2]))

            if config.SCHEDULER.NAME != 'one_cyle_lr':
                scheduler.step()

            if IS_LOCAL:
                utils.checkpoint.save_checkpoint(config, model, epoch, val_values[1], val_values[0])

            else:
                postfix['train_loss'] = train_values[0]
                postfix['train_res'] = train_values[1]
                postfix['train_acc'] = train_values[2]
                postfix['train_sens'] = train_values[3]
                postfix['train_spec'] = train_values[4]

                postfix['val_loss'] = val_values[0]
                postfix['val_res'] = val_values[1]
                postfix['val_acc'] = val_values[2]
                postfix['val_sens'] = val_values[3]
                postfix['val_spec'] = val_values[4]

                nsml.report(**postfix, summary=True, step=epoch)

                val_res = '%.10f' % val_values[1]
                val_res = val_res.replace(".", "")
                val_res = val_res[:4] + '.' + val_res[4:8] + '.' + val_res[8:]
                save_name = 'epoch_%02d_score%s_loss%.4f.pth' % (epoch, val_res, val_values[0])
                # nsml.save(save_name)
                nsml.save(epoch)

        for e, val_acc in val_acc_list:
            print('%02d %s' % (e, val_acc))


if __name__ == '__main__':
    main()
