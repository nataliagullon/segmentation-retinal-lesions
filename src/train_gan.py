import numpy as np
from src.gan_utils import TrainBatchFetcher, imgs2discr, imgs2gan, get_data
import os
from os.path import join
from model_gan import get_discriminator, get_gan, compile_unet
from utils.params import parse_arguments_gan, default_params


path_to_data = '../'


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


def main(**params):
    """
    It applies adversarial training to the previously trained segmentation model
    :param data_path: path where images and labels are located
    :param weights_path: path where weights of the u-net are saved for each iteration
    """

    params = dict(
        default_params,
        **params
    )
    verbose = params['verbose']

    train_path = join(path_to_data, join(params['data_path'], 'train/'))
    val_path = join(path_to_data, join(params['data_path'], 'val/'))

    if verbose:
        print("Getting models...")

    patch_size = params['patch_size']
    channels = params['channels']
    n_classes = params['n_classes']
    init_weights = join(path_to_data, join(params['weights_path'], params['init_weights']))
    if not init_weights.endswith('.h5'):
        init_weights = init_weights + '.h5'

    # generator
    g = compile_unet(patch_size, patch_size, channels, n_classes, init_weights)

    #discriminator
    d = get_discriminator(patch_size, patch_size, channels, n_classes)
    make_trainable(d, False)

    # adversarial
    gan = get_gan(g, d, patch_size, patch_size, channels, n_classes)

    n_rounds = 2 # number of rounds to apply adversarial training
    batch_size = params['batch_size']

    if verbose:
        print("Getting train and validation data...")

    # fetchers to train and validate the model
    images_train, labels_train, n_files_train = get_data(train_path, channels, n_classes, verbose, thres_score=0)
    train_fetcher = TrainBatchFetcher(images_train, labels_train, batch_size=batch_size)
    images_val, labels_val, n_files_val = get_data(val_path, channels, n_classes, verbose)
    val_fetcher = TrainBatchFetcher(images_val, labels_val, batch_size=batch_size)
    #steps_per_epoch_d = (n_files_train//batch_size +1)
    #steps_per_epoch_g = (n_files_train//batch_size +1)
    #steps_val_d = (n_files_val//batch_size +1)
    #steps_val_g = (n_files_val//batch_size +1)
    steps_per_epoch_d = 5
    steps_per_epoch_g = 5
    steps_val_d = 5
    steps_val_g = 5

    # to show the progression of the losses
    val_gan_loss_array = np.zeros(n_rounds)
    val_discr_loss_array = np.zeros(n_rounds)
    gan_loss_array = np.zeros(n_rounds)
    discr_loss_array = np.zeros(n_rounds)

    if verbose:
        print("Adversarial training...")

    weights_path = join(path_to_data, params['weights_path'])
    name_weights = params['weights']

    for n_round in range(n_rounds):
        print("******* TRAIN *******")

        # train D
        make_trainable(d, True)
        for i in range(steps_per_epoch_d):
            image_batch, labels_batch = next(train_fetcher)
            pred = g.predict(image_batch)
            img_discr_batch, lab_discr_batch = imgs2discr(image_batch, labels_batch, pred)
            loss, acc, dice = d.train_on_batch(img_discr_batch, lab_discr_batch)
        discr_loss_array[n_round] = loss
        print("DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss))

        # train GAN
        make_trainable(d, False)
        for i in range(steps_per_epoch_g):
            image_batch, labels_batch = next(train_fetcher)
            img_gan_batch, lab_gan_batch = imgs2gan(image_batch, labels_batch)
            loss, acc, dice = gan.train_on_batch(img_gan_batch, lab_gan_batch)
        gan_loss_array[n_round] = loss
        print("UNET Round: {0} -> Loss {1}".format((n_round + 1), loss))

        # evalutation on validation dataset
        print("******* VALIDATION *******")

        # D
        for i in range(steps_val_d):
            image_val_batch, labels_val_batch = next(val_fetcher)
            pred = g.predict(image_val_batch)
            img_discr_val, lab_discr_val = imgs2discr(image_val_batch, labels_val_batch, pred)
            loss_val, acc_val, dice_val = d.test_on_batch(img_discr_val, lab_discr_val)
        val_discr_loss_array[n_round] = loss_val
        print("DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round+1), loss_val))

        # GAN
        for i in range(steps_val_g):
            image_val_batch, labels_val_batch = next(val_fetcher)
            img_gan_val, lab_gan_val = imgs2gan(image_val_batch, labels_val_batch)
            loss_val, acc_val, dice_val = gan.test_on_batch(img_gan_val, lab_gan_val)
        val_gan_loss_array[n_round] = loss_val
        print("UNET Round: {0} -> Loss {1}".format((n_round + 1), loss_val))

        # save the weights of the unet
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        g.save_weights(os.path.join(weights_path, name_weights + "_{}.h5".format(n_round)))

    # print the evolution of the loss
    if verbose:
        print("\n********* Evolution of loss *********")
        print("DISCR loss {}".format(discr_loss_array))
        print("UNET loss {}".format(gan_loss_array))
        print("DISCR validation loss {}".format(val_discr_loss_array))
        print("UNET validation loss {}".format(val_gan_loss_array))

    file = weights_path + 'loss_evolution_adv_train_' + params['weights'] + '.txt'
    f = open(file, "a+")
    f.write('Loss evolution for adversarial training with weights ' + params['weights'] + '\n')

    f.write("DISCR loss {}\n".format(discr_loss_array))
    f.write("UNET loss {}\n".format(gan_loss_array))
    f.write("DISCR validation loss {}\n".format(val_discr_loss_array))
    f.write("UNET validation loss {}\n\n".format(val_gan_loss_array))

    f.close()


if __name__ == '__main__':
    main(**vars(parse_arguments_gan()))