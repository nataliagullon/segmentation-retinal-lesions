from utils import model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical

import numpy as np

from utils.gan import TrainBatchFetcher, imgs2discr, imgs2gan, get_data
import sys
import os

from utils.model_gan import get_discriminator, get_gan, compile_unet
from utils.model import get_unet
from evaluate import AUC_PR

patch_size = 400
channels = 3
out_channels = 1

n_epochs = 100
batch_size = 32
smooth = 0.01


labels = ['EX/', 'HE/', 'MA/', 'SE/']


def make_trainable(net, val):
    """
    If False, it fixes the network and it is not trainable (the weights are frozen)
    If True, the network is trainable (the weights can be updated)
    :param net: network
    :param val: boolean to make the network trainable or not
    """
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def main(data_path, weights_path):
    """
    It applies adversarial training to the previously trained segmentation model
    :param data_path: path where images and labels are located
    :param weights_path: path where weights of the u-net are saved for each iteration
    """

    train_path = data_path + '400_train/'
    val_path = data_path + '400_val/'

    print "Getting models..."
    # generator
    #g = get_unet(400, 400, 3, 5)
    g = compile_unet(400, 400, 3, 5)

    #discriminator
    d = get_discriminator(400, 400, 3, 5)
    make_trainable(d, False)

    # adversarial
    gan = get_gan(g, d, 400, 400, 3, 5)

    n_rounds = 20 # number of rounds to apply adversarial training
    batch_size = 8

    print "Getting train and validation data..."
    # fetchers to train and validate the model
    images_train, labels_train, n_files_train = get_data(train_path, thres_score=0)
    train_fetcher = TrainBatchFetcher(images_train, labels_train, batch_size=batch_size)
    images_val, labels_val, n_files_val = get_data(val_path)
    val_fetcher = TrainBatchFetcher(images_val, labels_val, batch_size=batch_size)
    steps_per_epoch_d = (n_files_train//batch_size +1)
    steps_per_epoch_g = (n_files_train//batch_size +1)
    steps_val_d = (n_files_val//batch_size +1)
    steps_val_g = (n_files_val//batch_size +1)

    # to show the progression of the losses
    val_gan_loss_array = np.zeros(n_rounds)
    val_discr_loss_array = np.zeros(n_rounds)
    gan_loss_array = np.zeros(n_rounds)
    discr_loss_array = np.zeros(n_rounds)

    print "Adversarial training..."

    for n_round in range(n_rounds):
        print "******* TRAIN *******"
        # train D
        make_trainable(d, True)
        for i in range(steps_per_epoch_d):
            image_batch, labels_batch = next(train_fetcher)
            pred = g.predict(image_batch)
            img_discr_batch, lab_discr_batch = imgs2discr(image_batch, labels_batch, pred)
            loss, acc, dice = d.train_on_batch(img_discr_batch, lab_discr_batch)
        discr_loss_array[n_round] = loss
        print "DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss)

        # train GAN
        make_trainable(d, False)
        for i in range(steps_per_epoch_g):
            image_batch, labels_batch = next(train_fetcher)
            img_gan_batch, lab_gan_batch = imgs2gan(image_batch, labels_batch)
            loss, acc, dice = gan.train_on_batch(img_gan_batch, lab_gan_batch)
        gan_loss_array[n_round] = loss
        print "GAN Round: {0} -> Loss {1}".format((n_round + 1), loss)

        # evalutation on validation dataset
        print "******* VALIDATION *******"

        # D
        for i in range(steps_val_d):
            image_val_batch, labels_val_batch = next(val_fetcher)
            pred = g.predict(image_val_batch)
            img_discr_val, lab_discr_val = imgs2discr(image_val_batch, labels_val_batch, pred)
            loss_val, acc_val, dice_val = d.test_on_batch(img_discr_val, lab_discr_val)
        val_discr_loss_array[n_round] = loss_val
        print "DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss_val)

        # GAN
        for i in range(steps_val_g):
            image_val_batch, labels_val_batch = next(val_fetcher)
            img_gan_val, lab_gan_val = imgs2gan(image_val_batch, labels_val_batch)
            loss_val, acc_val, dice_val = gan.test_on_batch(img_gan_val, lab_gan_val)
        val_gan_loss_array[n_round] = loss_val
        print "GAN Round: {0} -> Loss {1}".format((n_round + 1), loss_val)

        # save the weights of the unet
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        g.save_weights(os.path.join(weights_path, "weights_{}.h5".format(n_round)))

    # print the evolution of the loss
    print "DISCR loss {}".format(discr_loss_array)
    print "GAN loss {}".format(gan_loss_array)
    print "DISCR validation loss {}".format(val_discr_loss_array)
    print "GAN validation loss {}".format(val_gan_loss_array)


if __name__ == '__main__':
    data_path = '/imatge/ngullon/work/retina_data/'
    weights_path = '/imatge/ngullon/MyProjects/keras-unet/weights_gan_7/'
    main(data_path, weights_path)