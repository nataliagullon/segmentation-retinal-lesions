import os
from os.path import join
from model_unet import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from unet_generator import UNetGeneratorClass
from utils.losses import *
from utils.params import parse_arguments_w, default_params
import sys


path_to_data = '../'

def train_unet_generator(**params):
    """
    It trains the U-Net first using patches with a percentage (thres_score) of lesion and after using patches with lesion.
    It saves the best weights using model checkpointer
    :param data_path: path where images and labels are located
    :param out_path: path where the evolution of the performance (TensorBorad) is saved
    """

    params = dict(
        default_params,
        **params
    )
    verbose = params['verbose']

    if verbose:
        print("Welcome to U-Net training")

    out_path = join(path_to_data, params['tensorboard_unet'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if os.listdir(out_path):
            os.system('rm ' + out_path + '*')

    if verbose:
        print("Getting model...")

    patch_size = params['patch_size']
    channels = params['channels']
    n_classes = params['n_classes']
    unet = get_unet(patch_size, patch_size, channels, n_classes)

    metrics = [generalised_dice_coef]
    lr = params['lr']
    loss = gen_dice_multilabel

    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)

    data_path = join(path_to_data, params['data_path'])
    batch_size = params['batch_size']

    if verbose:
        print("Getting generators...")

    train_scored_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                                channels=channels, apply_augmentation=False, thres_score=0.3, train=True)

    train_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                         channels=channels, apply_augmentation=True, thres_score=0, train=True)

    val_generator = UNetGeneratorClass(data_path=data_path, n_class=n_classes, batch_size=batch_size,
                                       channels=channels, apply_augmentation=False, thres_score=None, train=False)

    if verbose:
        print("Files in scored generator for training:", len(train_scored_generator.files))
        print("Training model...")

    weights_path = join(path_to_data, params['weights_unet'])
    best_weights = join(weights_path, params['weights'] + '.h5')

    model_checkpoint = ModelCheckpoint(best_weights, verbose=1, monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=out_path, histogram_freq=0, write_graph=True, write_images=False)

    unet.fit_generator(train_scored_generator.generate(),
                       steps_per_epoch=(len(train_scored_generator.files) // train_scored_generator.batch_size + 1),
                       epochs=15, verbose=verbose)

    unet.fit_generator(train_generator.generate(),
                       steps_per_epoch=(len(train_generator.files) // train_generator.batch_size + 1),
                       epochs=50, verbose=verbose, callbacks=[tensorboard, model_checkpoint],
                       validation_data=val_generator.generate(),
                       validation_steps=(len(val_generator.files) // val_generator.batch_size + 1))


    #unet.save_weights(join(weights_path, 'last_weights.h5'), overwrite=True)

    if verbose:
        print("Training finished")


if __name__ == '__main__':
    train_unet_generator(**vars(parse_arguments_w()))