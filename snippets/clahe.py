import cv2
import numpy as np


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


def ben(img, sigmaX=30):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)


img = cv2.imread('000000_01.jpg')
img = cv2.resize(img, (512,384))


img_clahe = clahe(img)
img_ben = ben(img)
img_clahe_ben = ben(clahe(img))
img_ben_clahe = clahe(ben(img))


cv2.imshow('img', img)
cv2.imshow('img_clahe', img_clahe)
cv2.imshow('img_ben', img_ben)
cv2.imshow('img_clahe_ben', img_clahe_ben)
cv2.imshow('img_ben_clahe', img_ben_clahe)

cv2.imwrite('img_clahe.jpg', img_clahe)
cv2.imwrite('img_ben.jpg', img_ben)
cv2.imwrite('img_clahe_ben.jpg', img_clahe_ben)
cv2.imwrite('img_ben_clahe.jpg', img_ben_clahe)

cv2.waitKey()