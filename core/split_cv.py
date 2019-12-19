import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


#  label 참고용
def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    if label == 'AMD':		cls = 1;    resvec[cls] = 1
    elif label == 'RVO':	cls = 2;    resvec[cls] = 1
    elif label == 'DMR':	cls = 3;    resvec[cls] = 1
    else:					cls = 0;    resvec[cls] = 1		# Normal

    return resvec


def split_cv(root_path, n_splits=10):
    df = pd.DataFrame(columns=['fpath', 'disease', 'label', 'fold'])

    fpaths = []
    diseases = []
    labels = []

    dir_names = ['NOR', 'AMD', 'RVO', 'DMR']

    for i, dir_name in enumerate(dir_names):
        dir_fpaths = [os.path.join(dir_name, f) for f in os.listdir(os.path.join(root_path, dir_name))]

        fpaths += dir_fpaths

        diseases += [dir_name] * len(dir_fpaths)
        labels += [i] * len(dir_fpaths)

        print(dir_name, len(dir_fpaths))

    df['fpath'] = fpaths
    df['disease'] = diseases
    df['label'] = labels

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    skf.get_n_splits(fpaths, labels)

    for fold, (train_index, val_index) in enumerate(skf.split(fpaths, labels)):
        # print(fold, len(train_index), len(val_index))
        df['fold'].iloc[val_index] = fold

    df['label'] = df['label'].astype(int)
    df['fold'] = df['fold'].astype(int)

    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    return df
