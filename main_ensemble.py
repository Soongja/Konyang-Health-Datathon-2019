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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


import utils.config
from models.model_factory import get_model
from datasets.dataloader import get_dataloader
from core.preprocess import none, ben, clahe, ben_clahe, clahe_ben, preprocess_dict


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


ensemble_checkpoints = (
    ('team146/KHD2019_FUNDUS/80', '26', 'configs/ensemble_0.yml'),
    ('team146/KHD2019_FUNDUS/85', '30', 'configs/ensemble_1.yml'),
    ('team146/KHD2019_FUNDUS/86', '32', 'configs/ensemble_2.yml'),
    ('team146/KHD2019_FUNDUS/91', '21', 'configs/ensemble_3.yml'),
    ('team146/KHD2019_FUNDUS/104', '25', 'configs/ensemble_4.yml'),
    ('team146/KHD2019_FUNDUS/113', '27', 'configs/ensemble_5.yml'),
    ('team146/KHD2019_FUNDUS/119', '31', 'configs/ensemble_6.yml'),
    ('team146/KHD2019_FUNDUS/124', '33', 'configs/ensemble_7.yml'),
    ('team146/KHD2019_FUNDUS/136', '26', 'configs/ensemble_8.yml'),
    ('team146/KHD2019_FUNDUS/138', '23', 'configs/ensemble_9.yml'),
    ('team146/KHD2019_FUNDUS/145', '31', 'configs/ensemble_10.yml'),
    # ('team146/KHD2019_FUNDUS/161', '26', 'configs/ensemble_11.yml'),
    # ('team146/KHD2019_FUNDUS/165', '32', 'configs/ensemble_12.yml'),
    # ('team146/KHD2019_FUNDUS/166', '25', 'configs/ensemble_13.yml'),
    # ('team146/KHD2019_FUNDUS/170', '27', 'configs/ensemble_14.yml'),
    # ('team146/KHD2019_FUNDUS/186', '27', 'configs/ensemble_15.yml'),

    ('team146/KHD2019_FUNDUS/120', '21', 'configs/ensemble_sj_0.yml'),
    ('team146/KHD2019_FUNDUS/139', '26', 'configs/ensemble_sj_1.yml'),
)


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
        return _infer(model, data)

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

    config = utils.config.load(ensemble_checkpoints[0][2])
    model = get_model(config).cuda()
    bind_model(model)

    args = get_args()
    if args.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if args.mode == 'train':  ### training mode일 때
        print('Training Start...')

        nsml.load(session=ensemble_checkpoints[0][0], checkpoint=ensemble_checkpoints[0][1])
        nsml.save(0)
        exit()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class TestDataset(Dataset):
    def __init__(self, data, config, transform=None):
        self.data = data
        self.config = config
        self.transform = transform

        print('test set:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ??

        # if idx == 0:
            # print(image.shape)
            # print(image)
            # print('channel mean:', np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2]))

        image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        # # in case of runtime preprocessing
        preprocess_func, preprocess_norm = preprocess_dict[self.config.DATA.PREPROCESS]
        #
        # image = preprocess_func(image)

        if self.transform is not None:
            image = self.transform(image)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            preprocess_norm,
        ])
        image = normalize(image)

        return image


def get_test_loader(data, config, transform=None):
    dataset = TestDataset(data, config, transform=transform)

    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)

    return dataloader


class VFlip:
    def __call__(self, image):
        return np.flip(image, axis=0).copy()
        # return image[::-1]


class HFlip:
    def __call__(self, image):
        return np.flip(image, axis=1).copy()
        # return image[:,::-1]


def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        # start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)

            preds = F.softmax(logits, dim=1)
            preds = preds.detach().cpu().numpy()

            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

    output = np.concatenate(output, axis=0)
    return output


def run(model, data, config):
    ####################################################################################################
    test_loader = get_test_loader(data, config, transform=None)
    out = inference(model, test_loader)
    # print(out)
    ####################################################################################################
    # print('Vflip')
    # test_loader = get_test_loader(data, config, transform=VFlip())
    # out_vflip = inference(model, test_loader)
    # print(out_vflip)
    # out += out_vflip
    # del out_vflip
    ####################################################################################################
    print('Hflip')
    test_loader = get_test_loader(data, config, transform=HFlip())
    out_hflip = inference(model, test_loader)
    # print(out_hflip)
    out += out_hflip
    del out_hflip
    ####################################################################################################

    out = out / 2.0
    # print(out.shape)
    # print(out)

    # out = np.argmax(out, axis=1)
    # print(out.shape)
    # print(out)

    return out


def _infer(model, data):
    start = time.time()
    ################################################################################
    print('test preprocessing start!')
    # # data: [a, b, c,...]
    data_bc = []
    bc_func, _ = preprocess_dict["ben_clahe"]

    for d in data:
        d = cv2.resize(d, (704,544))
        data_bc.append(bc_func(d))
        # del d

    # del data
    ellapsed = time.time() - start
    print('test preprocessing time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
    print('test preprocessing ended!')
    del data

    ################################################################################

    # n_ensemble = len(ensemble_checkpoints)
    final = []

    for sess, ckpt, config_path in ensemble_checkpoints:
        config = utils.config.load(config_path)

        model = get_model(config).cuda()
        bind_model(model)

        nsml.load(checkpoint=ckpt, session=sess)

        # data_processed = []
        # _func, _ = preprocess_dict[config.DATA.PREPROCESS]
        # for d in data:
        #     d = cv2.resize(d, (config.DATA.IMG_W, config.DATA.IMG_H))
        #     data_processed.append(_func(d))

        out = run(model, data_bc, config)
        final.append(out)

        del model

    # final = sum(final) / float(n_ensemble)
    final = sum(final)

    final = np.argmax(final, axis=1)
    print(final.shape)
    print(final)

    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

    return final


if __name__ == '__main__':
    main()
