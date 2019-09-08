from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def get_unet(patch_height, patch_width, channels, n_classes):
    """
    It creates a U-Net and returns the model
    :param patch_height: height of the input images
    :param patch_width: width of the input images
    :param channels: channels of the input images
    :param n_classes: number of classes
    :return: the model (unet)
    """
    axis = 3
    k = 3 # kernel size
    s = 2 # stride
    n_filters = 32 # number of filters

    inputs = Input((patch_height, patch_width, channels))
    conv1 = Conv2D(n_filters, (k,k), padding='same')(inputs)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding='same')(conv1)
    conv1 = BatchNormalization(scale=False, axis=axis)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s,s))(conv1)

    conv2 = Conv2D(2*n_filters, (k,k), padding='same')(pool1)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding='same')(conv2)
    conv2 = BatchNormalization(scale=False, axis=axis)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s,s))(conv2)

    conv3 = Conv2D(4*n_filters, (k,k), padding='same')(pool2)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding='same')(conv3)
    conv3 = BatchNormalization(scale=False, axis=axis)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(pool3)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding='same')(conv4)
    conv4 = BatchNormalization(scale=False, axis=axis)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(pool4)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding='same')(conv5)
    conv5 = BatchNormalization(scale=False, axis=axis)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k,k), padding='same')(up1)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding='same')(conv6)
    conv6 = BatchNormalization(scale=False, axis=axis)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(up2)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding='same')(conv7)
    conv7 = BatchNormalization(scale=False, axis=axis)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(up3)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding='same')(conv8)
    conv8 = BatchNormalization(scale=False, axis=axis)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=axis)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding='same')(up4)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding='same')(conv9)
    conv9 = BatchNormalization(scale=False, axis=axis)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(conv9)

    unet = Model(inputs=inputs, outputs=outputs)

    return unet