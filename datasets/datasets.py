import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from core.preprocess import none, ben, clahe, ben_clahe, clahe_ben, preprocess_dict


class FundusDataset(Dataset):
    def __init__(self, config, data_dir, fold_df, val_fold_idx, split, transform=None):
        self.config = config
        self.data_dir = data_dir
        self.transform = transform

        # columns: ['fpath', 'disease', 'label', 'fold']
        if split == 'train':
            self.fold_df = fold_df.loc[fold_df['fold'] != val_fold_idx].reset_index(drop=True)
        else:
            self.fold_df = fold_df.loc[fold_df['fold'] == val_fold_idx].reset_index(drop=True)

        if split == 'train' and self.config.DEBUG:
            self.fold_df = self.fold_df[:400]
        print(split, 'set:', len(self.fold_df))

        self.labels = self.fold_df['label'].values

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        f_path = self.fold_df["fpath"][idx]
        image = cv2.imread(os.path.join(self.data_dir, f_path), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        # in case of runtime preprocessing
        _, preprocess_norm = preprocess_dict[self.config.DATA.PREPROCESS]

        # image = preprocess_func(image)

        if self.transform is not None:
            image = self.transform(image)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            preprocess_norm,
        ])

        image = normalize(image)

        label = self.fold_df["label"][idx]
        label = torch.LongTensor([label])

        return image, label
