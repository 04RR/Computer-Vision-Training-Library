import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import os
import time
import random
import torchvision.transforms as ttf
import cv2
import matplotlib.pylab as plt


# WRONG 
def get_overlapping_grids(img, kernal_size, stride):
    
    h, w, _ = img.shape
    csteps = int((h / stride) - 1)
    rsteps = int((w / stride) - 1)


    crops = []
    img_full = np.random.randn(csteps * kernal_size, rsteps * kernal_size, 3)

    for i in range(csteps):
        for j in range(rsteps):
            crop = np.array(
                img[
                    stride * i : stride * i + kernal_size,
                    stride * j : stride * j + kernal_size,
                    :,
                ]
            )
            
            img_full[
                kernal_size * i : kernal_size * i + kernal_size,
                kernal_size * j : kernal_size * j + kernal_size,
                :,
            ] = crop

            crops.append(crop)

    return crops, img_full


def read_image(filename, resize=False):

    image = cv2.imread(filename)

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize:
        image = cv2.resize(image, resize)

    return image


def display_images(images, nrows=3, ncols=3, cmap=None, title=None):
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    if title:
        fig.suptitle(title, fontsize=20)
    for i in range(ncols):
        for j in range(nrows):
            ax[i][j].imshow(images[i], cmap=cmap)
            ax[i][j].axis("off")
    plt.show()


img = read_image("D:\\Desktop\\test.jpeg")
grids, full = get_overlapping_grids(img, kernal_size=100, stride=100)
display_images(grids)
plt.imshow(full)
plt.show()