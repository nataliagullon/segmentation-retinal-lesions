from utils import model

from keras.optimizers import Adam, RMSprop
from keras.models import load_model

from skimage.io import imsave
from skimage.io import imread
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from train import dice_coef, dice_coef_multilabel, generalised_dice_coef, gen_dice_multilabel
from evaluate import AUC_PR

import sys

data_path = '/imatge/ngullon/work/retina_data/'
labels = ['EX/', 'HE/', 'MA/', 'SE/']

color_code_labels = [
    [0, 0, 0],      # 0 - Black     - Background
    [255, 0, 0],    # 1 - Red       - EX class
    [0, 255, 0],    # 2 - Green     - HE class
    [0, 0, 255],    # 3 - Blue      - MA class
    [255, 255, 0],  # 4 - Yellow    - SE class
]

num_patches = 6


def main(patch_size_test, path_test):
    """
    It predicts the test images and evaluates the model using AUC precision-recall.
    The images are predicted first with patches and then they are reconstructed.
    """
    print "Getting model..."
    unet = model.get_unet(1600, 1600, 3, 5)

    metrics = ['accuracy', dice_coef]

    unet.compile(optimizer=Adam(lr=1e-4), loss=gen_dice_multilabel, metrics=metrics)
    print "Loading weights..."
    unet.load_weights('weights_gan_6/weights_2.h5')

    print "Predicting..."

    path_predicted_patches = path_test + 'predictions/pred_patches/'
    if not os.path.exists(path_predicted_patches):
        os.makedirs(path_predicted_patches)

    test_images_names = []

    path_test_patches = path_test + 'images/'
    files_test = [f for f in listdir(path_test_patches) if isfile(join(path_test_patches, f))]

    # AUC
    gt_ex = np.zeros((len(files_test), patch_size_test, patch_size_test))
    pred_ex = np.zeros((len(files_test), patch_size_test, patch_size_test))
    gt_he = np.zeros((len(files_test), patch_size_test, patch_size_test))
    pred_he = np.zeros((len(files_test), patch_size_test, patch_size_test))
    gt_ma = np.zeros((len(files_test), patch_size_test, patch_size_test))
    pred_ma = np.zeros((len(files_test), patch_size_test, patch_size_test))
    gt_se = np.zeros((len(files_test), patch_size_test, patch_size_test))
    pred_se = np.zeros((len(files_test), patch_size_test, patch_size_test))

    # patches prediction
    i = 0
    for file in files_test:
        img_name = file[:-4]
        full_image_name = file[:-8]
        if full_image_name not in test_images_names:
            test_images_names.append(full_image_name)
        image = imread(path_test_patches + file)
        img_array = np.asarray(image).astype(np.uint8)
        img_array = np.expand_dims(img_array, axis=0)
        pred = unet.predict(img_array, verbose = 1)
        pred_prob = pred[0]
        pred = np.argmax(pred[0], axis=2)

        print "Saving predicted image..."
        img_pred = np.zeros(pred.shape + (3,), dtype=np.uint8)
        for ind in range(len(color_code_labels)):
            img_pred[pred == ind, :] = color_code_labels[ind]

        label_pred = np.zeros(pred.shape + (3,), dtype=np.uint8)
        label_pred[:,:,0] = pred

        imsave(path_predicted_patches + img_name + '_pred.png', img_pred)
        imsave(path_predicted_patches + img_name + '_lab_pred.png', label_pred)

        # AUC PR
        a,b,c = file.split("_")
        ex_name = a + '_' + b + '_EX_' + c[:-4] + '.tif'
        ex_file = imread(path_test + 'EX_patches/' + ex_name)
        ex_file = np.asarray(ex_file).astype(np.uint8)
        ex_image = ex_file[:, :, 0]/255
        gt_ex[i, :, :] = ex_image
        pred_ex[i, :, :] = pred_prob[:, :, 1]

        he_name = a + '_' + b + '_HE_' + c[:-4] + '.tif'
        he_file = imread(path_test + 'HE_patches/' + he_name)
        he_file = np.asarray(he_file).astype(np.uint8)
        he_image = he_file[:, :, 0] / 255
        gt_he[i, :, :] = he_image
        pred_he[i, :, :] = pred_prob[:, :, 2]

        ma_name = a + '_' + b + '_MA_' + c[:-4] + '.tif'
        ma_file = imread(path_test + 'MA_patches/' + ma_name)
        ma_file = np.asarray(ma_file).astype(np.uint8)
        ma_image = ma_file[:, :, 0] / 255
        gt_ma[i, :, :] = ma_image
        pred_ma[i, :, :] = pred_prob[:, :, 3]

        se_name = a + '_' + b + '_SE_' + c[:-4] + '.tif'
        se_file = imread(path_test + 'SE_patches/' + se_name)
        se_file = np.asarray(se_file).astype(np.uint8)
        se_image = se_file[:, :, 0] / 255
        gt_se[i, :, :] = se_image
        pred_se[i, :, :] = pred_prob[:, :, 4]

        i += 1
    print "Prediction done"

    # print AUC PR
    auc_ex = AUC_PR(gt_ex, pred_ex)
    print "AUC Precision-Recall for class EX: {}".format(auc_ex)
    auc_he = AUC_PR(gt_he, pred_he)
    print "AUC Precision-Recall for class HE: {}".format(auc_he)
    auc_ma = AUC_PR(gt_ma, pred_ma)
    print "AUC Precision-Recall for class MA: {}".format(auc_ma)
    auc_se = AUC_PR(gt_se, pred_se)
    print "AUC Precision-Recall for class SE: {}".format(auc_se)

    path_predicted_images = path_test + 'predictions/pred_images/'
    if not os.path.exists(path_predicted_images):
        os.makedirs(path_predicted_images)

    # reconstruction of images from the predicted patches
    print "Reconstructing predicted images..."
    for file in test_images_names:
        full_image = np.zeros((3200, 4800, 3), dtype=np.uint8)
        full_label = np.zeros((3200, 4800, 3), dtype=np.uint8)

        for i in xrange(num_patches):
            patch_name = file + '_' + str(i+1).zfill(3) + '_pred.png'
            patch_lab_name = file + '_' + str(i + 1).zfill(3) + '_lab_pred.png'

            patch = imread(path_predicted_patches + patch_name)
            lab_patch = imread(path_predicted_patches + patch_lab_name)

            if i < 3:
                full_image[0:1600, i*1600:(i+1)*1600, :] = patch
                full_label[0:1600, i*1600:(i+1)*1600, :] = lab_patch
            else:
                full_image[1600:(2*1600), (i-3)*1600:(i-2)*1600, :] = patch
                full_label[1600:(2 * 1600), (i - 3) * 1600:(i - 2) * 1600, :] = lab_patch

        image_labels = full_image[:, 0:4400, :]
        image_labels_norm = full_label[:, 0:4400, :]

        imsave(path_predicted_images + file + '_pred.png', image_labels)
        imsave(path_predicted_images + file + '_lab_pred.png', image_labels_norm)

        print "Predictions of {} saved".format(file)


if __name__ == '__main__':
    main(1600, data_path + '400_test/')