import os
import time
import glob
import cv2
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
from functools import partial
import multiprocessing as mp


def none(img):
    return img


def ben(img, sigmaX=30):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)


def clahe(img, clip_limit=4.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


def ben_clahe(img):
    img = ben(img)
    img = clahe(img)
    return img


def clahe_ben(img):
    img = clahe(img)
    img = ben(img)
    return img


preprocess_dict = {"ben": (ben, transforms.Normalize(mean=[0.55675373, 0.56194775, 0.5059191 ],
                                                     std=[0.26485795, 0.20592705, 0.03625965])),
                   "clahe": (clahe, transforms.Normalize(mean=[0.28994021, 0.30553625, 0.14231476],
                                                     std=[0.19999885, 0.18765197, 0.10710731])),
                   "ben_clahe": (ben_clahe, transforms.Normalize(mean=[0.51966318, 0.52400417, 0.46535194],
                                                     std=[0.28349236, 0.25107314, 0.13092929])),
                   "clahe_ben": (clahe_ben, transforms.Normalize(mean=[0.56271693, 0.56290443, 0.53868806],
                                                     std=[0.32947288, 0.3164882,  0.27737021])),
                   "none": (none, transforms.Normalize(mean=[0.17915059, 0.19568807, 0.00499657],
                                                     std=[0.1558116,  0.13737242, 0.0108313 ])),
                   }


def preprocess_multi(data_dir, output_dir, preprocess_type, cv2_size, fnames):
    for fname in fnames:
        img = cv2.imread(os.path.join(data_dir, fname))

        img = cv2.resize(img, cv2_size)

        preprocess_func, _ = preprocess_dict[preprocess_type]
        img = preprocess_func(img)

        cv2.imwrite(os.path.join(output_dir, fname), img)


def preprocess(data_dir, output_dir, preprocess_type='none', cv2_size=(192,160)):
    start = time.time()

    fnames = os.listdir(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_cpu = mp.cpu_count()
    pool = mp.Pool(n_cpu)
    n_cnt = len(fnames) // n_cpu

    fname_sets = [fnames[n_cnt * i:n_cnt * (i + 1)] for i in range(n_cpu)]
    fname_sets[-1] = fnames[n_cnt * (n_cpu - 1):]

    func = partial(preprocess_multi, data_dir, output_dir, preprocess_type, cv2_size)
    pool.map(func, fname_sets)
    pool.close()

    ellapsed = time.time() - start
    print('preprocessing finished in: %d hours %d minutes %d seconds' % (
            ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))


if __name__ == '__main__':
    preprocess('../data/train/AMD', '../data/train_ben_clahe/AMD')
    preprocess('../data/train/DMR', '../data/train_ben_clahe/DMR')
    preprocess('../data/train/NORMAL', '../data/train_ben_clahe/NORMAL')
    preprocess('../data/train/RVO', '../data/train_ben_clahe/RVO')
