import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from core.preprocess import none, ben, clahe, ben_clahe, clahe_ben, preprocess_dict


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

        if idx == 0:
            print(image.shape)
            print(image)
            print('channel mean:', np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2]))

        image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        # in case of runtime preprocessing
        preprocess_func, preprocess_norm = preprocess_dict[self.config.DATA.PREPROCESS]

        image = preprocess_func(image)

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
                            batch_size=16,
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


def _infer(model, data, config):
    ####################################################################################################
    test_loader = get_test_loader(data, config, transform=None)
    out = inference(model, test_loader)
    # print(out)
    ####################################################################################################
    print('Vflip')
    test_loader = get_test_loader(data, config, transform=VFlip())
    out_vflip = inference(model, test_loader)
    # print(out_vflip)
    out += out_vflip
    del out_vflip
    ####################################################################################################
    print('Hflip')
    test_loader = get_test_loader(data, config, transform=HFlip())
    out_hflip = inference(model, test_loader)
    # print(out_hflip)
    out += out_hflip
    del out_hflip
    ####################################################################################################

    out = out / 3.0
    print(out.shape)
    print(out)

    out = np.argmax(out, axis=1)
    print(out.shape)
    print(out)

    return out
