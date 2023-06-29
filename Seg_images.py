import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

img_path = os.getcwd() + "\\ISIC2018\\ISIC2018_Task1-2_Training_Input\\ISIC2018_Task1-2_Training_Input\\"
mask_path = os.getcwd() + "\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth\\ISIC2018_Task1_Training_GroundTruth\\"

width = 128
height = 128
channels = 3

train_img = glob(img_path + '*.jpg')
train_mask = [i.replace(img_path, mask_path).replace('.jpg', '_segmentation.png') for i in train_img]

# Display the first image and mask of the first subject.
image1 = np.array(Image.open(train_img[0]))
image1_mask = np.array(Image.open(train_mask[0]))
image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)

fig, ax = plt.subplots(3, 3, figsize=(8, 6))
for i in range(3):
    for j in range(3):
        if j == 0:
            ax[i, j].imshow(np.array(Image.open(train_img[i])), cmap='gray', aspect=True)

        elif j == 1:
            ax[i, j].imshow(np.array(Image.open(train_mask[i])), cmap='gray', aspect=True)
        elif j == 2:
            ax[i, j].imshow(np.array(Image.open(train_img[i])), cmap=None, interpolation='none')
            ax[i, j].imshow((np.ma.masked_where(Image.open(train_mask[i]) == 0, np.array(Image.open(train_mask[i])))),
                            cmap='jet', alpha=0.3)
        ax[i, j].axis('off')
        ax[0, 0].set_title('Input')
        ax[0, 1].set_title('Ground Truth')
        ax[0, 2].set_title('Segmented')

plt.show()
