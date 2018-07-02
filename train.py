from utils import model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from keras import backend as K

import numpy as np
import sys
import os
from unet_generator import UNetGeneratorClass

data_path = '/imatge/ngullon/work/retina_data/'
labels = ['EX/', 'HE/', 'MA/', 'SE/']


def generalised_dice_coef(y_true, y_pred, type_weight='Square'):
    """
    It computes the generalised dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :return: generalised dice coefficient score between y_true and y_pred
    """
    prediction = tf.cast(y_pred, tf.float32)

    ref_vol = tf.reduce_sum(y_true, axis=0)
    intersect = tf.reduce_sum(y_true * prediction, axis=0)
    seg_vol = tf.reduce_sum(prediction, axis = 0)

    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))

    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *tf.reduce_max(new_weights), weights)

    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator

    return generalised_dice_score


def gen_dice_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the generalised dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += generalised_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], type_weight='Square')
    return 1. - dice/5.


def dice_coef(y_true, y_pred, smooth = 1.):
    """
    It computes the dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param smooth: parameter to ensure stability
    :return: dice coefficient score between y_true and y_pred
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return 1. - dice/5.


def train_unet_generator(data_path, out_path):
    """
    It trains the U-Net first using patches with a percentage (thres_score) of lesion and after using patches with lesion.
    It saves the best weights using model checkpointer
    :param data_path: path where images and labels are located
    :param out_path: path where the evolution of the performance (TensorBorad) is saved
    """
    print "Welcome to U-Net training"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print "Getting model..."
    unet = model.get_unet(400, 400, 3, 5)

    metrics = [gen_dice_multilabel]

    unet.compile(optimizer=Adam(lr=1e-4), loss=gen_dice_multilabel, metrics=metrics)

    print "Getting generators..."
    train_scored_generator = UNetGeneratorClass(n_class=5, batch_size=8, apply_augmentation=False, thres_score=0.3,
                 path = data_path + '400_train/')

    train_generator = UNetGeneratorClass(n_class=5, batch_size=8, apply_augmentation=True, thres_score=0,
                 path = data_path + '400_train/')

    val_generator = UNetGeneratorClass(n_class=5, batch_size=8, apply_augmentation=False, thres_score=None,
                 path = data_path + '400_val/')

    print len(train_scored_generator.files)

    print "Training model..."
    model_checkpoint = ModelCheckpoint('best_weights_44.h5', verbose=1, monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=out_path, histogram_freq=0, write_graph=True, write_images=False)

    unet.fit_generator(train_scored_generator.generate(),
                       steps_per_epoch=(len(train_scored_generator.files) // train_scored_generator.batch_size + 1),
                       epochs=15, verbose=1, callbacks=[tensorboard])

    unet.fit_generator(train_generator.generate(),
                       steps_per_epoch=(len(train_generator.files) // train_generator.batch_size + 1),
                       epochs=50, verbose=1, callbacks=[tensorboard, model_checkpoint],
                       validation_data=val_generator.generate(),
                       validation_steps=(len(val_generator.files) // val_generator.batch_size + 1))

    unet.save_weights('last_weights_44.h5', overwrite=True)
    print "Training finished"


if __name__ == '__main__':
    data_path = '/imatge/ngullon/work/retina_data/'
    out_path = data_path + 'model_performance/performance_44/'
    train_unet_generator(data_path, out_path)