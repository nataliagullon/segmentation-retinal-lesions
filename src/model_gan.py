import numpy as np
import os

from keras import backend as K
from keras import objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as plot
from model import get_unet
from keras.utils import to_categorical
import tensorflow as tf
from train import gen_dice_multilabel, dice_coef_multilabel, dice_coef, generalised_dice_coef



def compile_unet(patch_height, patch_width, channels, n_classes):
    """
    It creates, compiles and loads the best weights of the previously trained U-Net
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the U-Net
    """
    unet = get_unet(patch_height, patch_width, channels, n_classes)

    unet.compile(optimizer=Adam(lr=1e-4), loss=gen_dice_multilabel, metrics=['accuracy', dice_coef])

    unet.load_weights('best_weights_37.h5')

    return unet


def get_discriminator(patch_height, patch_width, channels, n_classes):
    """
    It creates the discriminator, compiles and return the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the discriminator
    """

    k = 3  # kernel size
    s = 2  # stride
    n_filters = 32  # number of filters

    inputs = Input((patch_height, patch_width, channels + n_classes))
    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    gap = GlobalAveragePooling2D()(conv5)
    outputs = Dense(1, activation='sigmoid')(gap)

    d = Model(inputs, outputs)

    # loss of the discriminator. it is a binary loss
    def d_loss(y_true, y_pred):
        #L = objectives.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        L = objectives.mean_squared_error(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=1e-4), loss=d_loss, metrics=['accuracy', dice_coef])

    return d


def get_gan(g, d, patch_height, patch_width, channels, n_classes):
    """
    GAN (that binds generator and discriminator)
    It gets the combined and compiles model (U-Net + discriminator)
    :param g: segmentation model
    :param d: discriminative model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the combined model (U-Net + discriminator)
    """

    image = Input((patch_height, patch_width, channels))
    labels = Input((patch_height, patch_width, n_classes))

    fake_labels = g(image)
    fake_pair = Concatenate(axis=3)([image, fake_labels])

    gan = Model([image, labels], d(fake_pair))

    # loss of the combined model. it has a component that penalizes that the discriminator classify the outputs of the
    # unet as fake and another component that penalizes the difference between real and predicted segmentation maps
    def gan_loss(y_true, y_pred):

        #trade-off coefficient
        alpha_recip = 0.05

        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        #L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
        L_adv = objectives.mean_squared_error(y_true_flat, y_pred_flat)

        L_seg = gen_dice_multilabel(labels, fake_labels)

        return alpha_recip * L_adv + L_seg

    gan.compile(optimizer=Adam(lr=1e-4), loss=gan_loss, metrics=['accuracy', dice_coef])

    return gan


if __name__ == '__main__':
    unet = compile_unet(400, 400, 3, 5)
    d = get_discriminator(400, 400, 3, 5)
    gan = get_gan(unet, d, 400, 400, 3, 5)

    print(unet.summary())
    print(d.summary())
    print(gan.summary())