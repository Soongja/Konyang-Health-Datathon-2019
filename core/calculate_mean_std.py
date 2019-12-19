import numpy as np
import cv2
from tqdm import tqdm
import os
from core.preprocess import none, ben, clahe, ben_clahe, clahe_ben, preprocess_dict


#### mode train is just train folder std
#### mode all.. train and test and total mean, std 3 set..

def calculate_std(mode='train'):
    if mode == 'train':
        train_std()
    elif mode == 'all':
        mean_tot, std_tot, cnt = train_std()
        test_std(mean_tot, std_tot, cnt)


def train_std(train_dir="../data/train", preprocess_type='none', cv2_size=(192,160)):

    mean_tot = np.zeros(3)
    std_tot = np.zeros(3)

    cnt = 0
    print("train data search in RGB space")

    train_fnames = []
    for folder, subfolders, files in os.walk(train_dir):
        for file in files:
            filePath = os.path.join(folder, file)
            filePath = filePath.replace('\\', '/')
            train_fnames.append(filePath)

    print('start calculating mean, std')
    for fname in train_fnames:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ##########################################

        # image = cv2.resize(image, cv2_size)

        # preprocess_func, _ = preprocess_dict[preprocess_type]
        # image = preprocess_func(image)

        ##########################################

        image = image.reshape(-1, 3) / 255.

        # train #
        mean_tot += image.mean(axis=0)
        std_tot += (image ** 2).mean(axis=0)
        cnt += 1

    channel_avr_train = mean_tot / cnt
    channel_std_train = np.sqrt(std_tot / cnt - channel_avr_train ** 2)

    print("\ntrain_avr:" + str(channel_avr_train))
    print("train_std:" + str(channel_std_train))

    return mean_tot, std_tot, cnt


def test_std(mean_tot, std_tot, cnt):
    test_dir = "../../data/test_images"
    print("test data search")
    mean_tot_test = np.zeros(3)
    std_tot_test = np.zeros(3)
    cnt_test = 0

    test_fnames = []
    for folder, subfolders, files in os.walk(test_dir):
        for file in files:
            filePath = os.path.join(folder, file)
            filePath = filePath.replace('\\', '/')
            test_fnames.append(filePath)

    for fname in tqdm(test_fnames):
        image = cv2.imread(os.path.join(test_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(-1, 3) / 255.

        # total_add #
        mean_tot += image.mean(axis=0)
        std_tot += (image ** 2).mean(axis=0)
        cnt += 1
        # test #
        mean_tot_test += image.mean(axis=0)
        std_tot_test += (image ** 2).mean(axis=0)
        cnt_test += 1

    channel_avr_test = mean_tot_test / cnt_test
    channel_std_test = np.sqrt(std_tot_test / cnt_test - channel_avr_test ** 2)

    print("\navr_test:" +str(channel_avr_test))
    print("std_test:" +str(channel_std_test))

    channel_avr_total = mean_tot / cnt
    channel_std_total = np.sqrt(std_tot / cnt - channel_avr_total ** 2)

    print("\navr_total:" +str(channel_avr_total))
    print("std_total:" +str(channel_std_total))


if __name__ == '__main__':
    train_std()
