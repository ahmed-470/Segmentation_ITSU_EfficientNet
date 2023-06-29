import os

from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from keras.optimizers import Optimizer
from PIL import Image
import PySimpleGUI as sg
import numpy as np
from matplotlib import pyplot as plt
import cv2
from efficientnet.keras import EfficientNetB7
from glob import glob
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Conv2D, Conv2DTranspose, \
    MaxPooling2D, concatenate, LeakyReLU, Add
import tensorflow as tf
from keras import backend as K, Input


def main():
    img_path = os.getcwd() + "\\ISIC2018\\ISIC2018_Task1-2_Training_Input\\ISIC2018_Task1-2_Training_Input\\"
    mask_path = os.getcwd() + "\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth\\ISIC2018_Task1_Training_GroundTruth\\"

    width = 128
    height = 128
    channels = 3

    train_img = glob(img_path + '*.jpg')
    train_mask = [i.replace(img_path, mask_path).replace('.jpg', '_segmentation.png') for i in train_img]

    # It contains 2594 training samples
    img_files = np.zeros([2594, height, width, channels])
    mask_files = np.zeros([2594, height, width])

    csfont = {'fontname': 'Times New Roman'}

    # load img_dataset

    print('Reading ISIC image 2018')
    # Retrieving the images and their labels
    k = 0
    for a in train_img:
        image = cv2.imread(a)
        image = cv2.resize(image, (width, height))
        image = np.array(image)
        img_files[k, :, :, :] = image
        k = k + 1

    print('Reading ISIC 2018 image finished')
    print('\nReading ISIC mask 2018')
    # Retrieving the images and their labels
    k = 0
    for b in train_mask:
        mask = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height))
        mask = np.array(mask)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask_files[k, :, :] = mask
        k = k + 1

    print('Reading ISIC 2018 mask finished')

    def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def dice_coef(y_true, y_pred):
        smooth = 0.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def iou(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
        return intersection / union

    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation == True:
            x = LeakyReLU(alpha=0.1)(x)
        return x

    def residual_block(blockInput, num_filters=16):
        x = LeakyReLU(alpha=0.1)(blockInput)
        x = BatchNormalization()(x)
        blockInput = BatchNormalization()(blockInput)
        x = convolution_block(x, num_filters, (3, 3))
        x = convolution_block(x, num_filters, (3, 3), activation=False)
        x = Add()([x, blockInput])
        return x

    def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.1):

        backbone = EfficientNetB7(weights='imagenet',
                                  include_top=False,
                                  input_shape=input_shape)
        inputs = Input(input_shape)

        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return Model(inputs=[inputs], outputs=[conv10])

    mask_files = mask_files[:, :, :, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(img_files, mask_files, test_size=0.2)

    class TunaSwarm(Optimizer):

        def __init__(self, epoch=10000, pop_size=100, **kwargs):
            """
            Args:
                epoch (int): maximum number of iterations, default = 10000
                pop_size (int): number of population size, default = 100
            """
            super().__init__(**kwargs)
            self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
            self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
            self.set_parameters(["epoch", "pop_size"])
            self.P = np.arange(1, 361)
            self.nfe_per_epoch = self.pop_size
            self.sort_flag = True

        def initialize_variables(self):
            self.aa = 0.7
            self.zz = 0.05

        def get_new_local_pos__(self, C, a1, a2, t, epoch):
            if np.random.rand() < self.zz:
                local_pos = self.generate_position(self.problem.lb, self.problem.ub)
            else:
                if np.random.rand() < 0.5:
                    r1 = np.random.rand()
                    beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((self.epoch - epoch) / self.epoch)))) * np.cos(
                        2 * np.pi * r1)
                    if np.random.rand() < C:
                        local_pos = a1 * (self.g_best[self.ID_POS] + beta * np.abs(
                            self.g_best[self.ID_POS] - self.pop[0][self.ID_POS])) + \
                                    a2 * self.pop[0][self.ID_POS]  # Equation (8.3)
                    else:
                        rand_pos = self.generate_position(self.problem.lb, self.problem.ub)
                        local_pos = a1 * (rand_pos + beta * np.abs(rand_pos - self.pop[0][self.ID_POS])) + a2 * \
                                    self.pop[0][self.ID_POS]  # Equation (8.1)
                else:
                    tf = np.random.choice([-1, 1])
                    if np.random.rand() < 0.5:
                        local_pos = tf * t ** 2 * self.pop[0][self.ID_POS]  # Eq 9.2
                    else:
                        local_pos = self.g_best[self.ID_POS] + np.random.rand(self.problem.n_dims) * (
                                self.g_best[self.ID_POS] - self.pop[0][self.ID_POS]) + \
                                    tf * t ** 2 * (self.g_best[self.ID_POS] - self.pop[0][self.ID_POS])
            return local_pos

        def evolve(self, epoch):
            """
            The main operations (equations) of algorithm. Inherit from Optimizer class

            Args:
                epoch (int): The current iteration
            """
            C = (epoch + 1) / self.epoch
            a1 = self.aa + (1 - self.aa) * C
            a2 = (1 - self.aa) - (1 - self.aa) * C
            t = (1 - (epoch + 1) / self.epoch) ** ((epoch + 1) / self.epoch)

            pop_new = []
            for idx in range(0, self.pop_size):
                if idx == 0:
                    pos_new = self.get_new_local_pos__(C, a1, a2, t, epoch)
                else:
                    if np.random.rand() < self.zz:
                        pos_new = self.generate_position(self.problem.lb, self.problem.ub)
                    else:
                        if np.random.rand() > 0.5:
                            r1 = np.random.rand()
                            beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * (self.epoch - epoch) / self.epoch))) * np.cos(
                                2 * np.pi * r1)
                            if np.random.rand() < C:
                                pos_new = a1 * (self.g_best[self.ID_POS] + beta * np.abs(
                                    self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])) + \
                                          a2 * self.pop[idx - 1][self.ID_POS]  # Eq. 8.4
                            else:
                                rand_pos = self.generate_position(self.problem.lb, self.problem.ub)
                                pos_new = a1 * (rand_pos + beta * np.abs(rand_pos - self.pop[idx][self.ID_POS])) + a2 * \
                                          self.pop[idx - 1][self.ID_POS]  # Eq 8.2
                        else:
                            tf = np.random.choice([-1, 1])
                            if np.random.rand() < 0.5:
                                pos_new = self.g_best[self.ID_POS] + \
                                          np.random.rand(self.problem.n_dims) * (
                                                  self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                                          tf * t ** 2 * (
                                                  self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])  # Eq 9.1
                            else:
                                pos_new = tf * t ** 2 * self.pop[idx][self.ID_POS]  # Eq 9.2
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            self.pop = self.update_target_wrapper_population(pop_new)

    model = UEfficientNet(input_shape=(height, width, channels))

    model.compile(optimizer='adam', loss='mse',
                  metrics=[iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef, sensitivity,
                           specificity])

    history = model.fit(X_train,
                        y_train,
                        epochs=200,
                        validation_data=(X_test, y_test))

    test_results = model.evaluate(X_test, y_test)
    test_metrics = np.array(test_results)

    metric_tr_iou = np.array(history.history['iou'])
    metric_tr_dc = np.array(history.history['dice_coef'])
    metric_tr_rec = np.array(history.history['recall'])
    metric_tr_pre = np.array(history.history['precision'])
    metric_tr_spe = np.array(history.history['specificity'])
    metric_tr_loss = np.array(history.history['loss'])
    metric_tr_sen = np.array(history.history['sensitivity'])

    train_metrics = np.column_stack(
        [metric_tr_loss, metric_tr_iou, metric_tr_rec, metric_tr_pre, metric_tr_dc, metric_tr_sen, metric_tr_spe])

    metric_val_iou = np.array(history.history['val_iou'])
    metric_val_dc = np.array(history.history['val_dice_coef'])
    metric_val_rec = np.array(history.history['val_recall'])
    metric_val_pre = np.array(history.history['val_precision'])
    metric_val_spe = np.array(history.history['val_specificity'])
    metric_val_loss = np.array(history.history['val_loss'])
    metric_val_sen = np.array(history.history['val_sensitivity'])

    validation_metrics = np.column_stack(
        [metric_val_loss, metric_val_iou, metric_val_rec, metric_val_pre, metric_val_dc, metric_val_sen,
         metric_val_spe])

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
                ax[i, j].imshow(
                    (np.ma.masked_where(Image.open(train_mask[i]) == 0, np.array(Image.open(train_mask[i])))),
                    cmap='jet', alpha=0.3)
            ax[i, j].axis('off')
            ax[0, 0].set_title('Input')
            ax[0, 1].set_title('Ground Truth')
            ax[0, 2].set_title('Segmented')

    plt.show()


# np.save('train_ISIC2018.npy', train_metrics)
# np.save('test_ISIC2018.npy', test_metrics)
# np.save('validation_ISIC2018.npy', validation_metrics)


VVV = sg.PopupYesNo('Do You want Complete Execution?')
if VVV == "Yes":
    main()
else:
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
                ax[i, j].imshow(
                    (np.ma.masked_where(Image.open(train_mask[i]) == 0, np.array(Image.open(train_mask[i])))),
                    cmap='jet', alpha=0.3)
            ax[i, j].axis('off')
            ax[0, 0].set_title('Input')
            ax[0, 1].set_title('Ground Truth')
            ax[0, 2].set_title('Segmented')

    plt.show()
    train_ISIC2018 = np.load('train_ISIC2018.npy')
    test_ISIC2018 = np.load('test_ISIC20l8.npy')
    validation_ISIC2018 = np.load('validation_ISIC2018.npy')

    train_std = np.std(train_ISIC2018, axis=1)
    valid_std = np.std(validation_ISIC2018, axis=1)

    train_mean = np.mean(train_ISIC2018, axis=1)
    valid_mean = np.mean(validation_ISIC2018, axis=1)

    # df = pd.DataFrame({"Loss": validation_ISIC2018[0, :], "IOU": validation_ISIC2018[1, :],
    #                    "Recall": validation_ISIC2018[2, :], "Precision": validation_ISIC2018[3, :],
    #                    "Dice Coefficient": validation_ISIC2018[4, :], "Sensitivity": validation_ISIC2018[5, :],
    #                    "Specificity": validation_ISIC2018[6, :]})
    # df.to_csv("validation_ISIC2018.csv", index=True)
    #
    # df = pd.DataFrame({"Loss": train_ISIC2018[0, :], "IOU": train_ISIC2018[1, :],
    #                    "Recall": train_ISIC2018[2, :], "Precision": train_ISIC2018[3, :],
    #                    "Dice Coefficient": train_ISIC2018[4, :], "Sensitivity": train_ISIC2018[5, :],
    #                    "Specificity": train_ISIC2018[6, :]})
    # df.to_csv("train_ISIC2018.csv", index=True)

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[0, :], color='g')
    plt.plot(validation_ISIC2018[0, :], color='m')
    plt.ylabel('Loss', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='upper right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[1, :], color='g')
    plt.plot(validation_ISIC2018[1, :], color='m')
    plt.ylabel('IOU', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='upper left', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[2, :], color='g')
    plt.plot(validation_ISIC2018[2, :], color='m')
    plt.ylabel('Recall', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='lower right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[3, :], color='g')
    plt.plot(validation_ISIC2018[3, :], color='m')
    plt.ylabel('Precision', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='lower right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[4, :], color='g')
    plt.plot(validation_ISIC2018[4, :], color='m')
    plt.ylabel('Dice coefficient', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='lower right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[5, :], color='g')
    plt.plot(validation_ISIC2018[5, :], color='m')
    plt.ylabel('Sensitivity', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='lower right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # Create empty plot
    ax1.set_facecolor('#FCFCFC')
    plt.grid(color='w', linestyle='-.', linewidth=1)
    plt.plot(train_ISIC2018[6, :], color='g')
    plt.plot(validation_ISIC2018[6, :], color='m')
    plt.ylabel('Specificity', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel(' Epoch', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    prop = {'family': 'Times New Roman', 'size': 18}
    plt.legend(['train', 'validation'], loc='lower right', fancybox=True, prop=prop)
    plt.tight_layout()
    plt.show()

    fig1, axis = plt.subplots(figsize=(11, 6))  # Create empty plot
    axis.set_facecolor('#C2C2C2')
    barWidth = 0.12
    br1 = np.arange(5)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    legend = ['DeepCNN', 'UNET', 'UNET + EfficientNetB7', 'Proposed']  # Legends

    plt.bar(br1, test_ISIC2018[0, :], color='b', width=barWidth,
            edgecolor='k', label=legend[0], linewidth=1.0, alpha=0.8)
    plt.bar(br2, test_ISIC2018[1, :], color='g', width=barWidth,
            edgecolor='k', label=legend[1], linewidth=1.0, alpha=0.8)
    plt.bar(br3, test_ISIC2018[2, :], color='gold', width=barWidth,
            edgecolor='k', label=legend[2], linewidth=1.0, alpha=0.8)
    plt.bar(br4, test_ISIC2018[3, :], color='r', width=barWidth,
            edgecolor='k', label=legend[3], linewidth=1.0, alpha=0.8)
    # Adding Labels, Ticks, Title and Legend
    legend_prop = {'size': 12}
    plt.legend(ncol=1, fancybox=True, shadow=True, prop=legend_prop, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel('Values', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xlabel('Test Metrics', fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.xticks([0.175, 1.175, 2.175, 3.175, 4.175], ['IOU', 'Recall', 'Precision', 'Dice \ncoefficient', 'Specificity'],
               fontsize=13,
               fontname='Times New Roman', fontweight='bold')

    plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')
    plt.tight_layout()

    plt.show()
