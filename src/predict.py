from model_unet import get_unet
from keras.optimizers import Adam
from skimage.io import imsave
from skimage.io import imread
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from utils.losses import dice_coef, gen_dice_multilabel
from evaluate import AUC_PR
from utils.params import parse_arguments_pred, default_params
from data import compute_num_patches
import sys

path_to_data = '../'


def predict(**params):
    """
    It predicts the test images and evaluates the model using AUC precision-recall.
    The images are predicted first with patches and then they are reconstructed.
    """

    params = dict(
        default_params,
        **params
    )

    verbose = params['verbose']

    if verbose:
        print("Getting model...")

    patch_size = params['patch_size_test']
    channels = params['channels']
    n_classes = params['n_classes']
    color_code_labels = params['color_code_labels']

    unet = get_unet(patch_size, patch_size, channels, n_classes)

    metrics = ['accuracy', dice_coef]
    lr = params['lr']

    unet.compile(optimizer=Adam(lr=lr), loss=gen_dice_multilabel, metrics=metrics)

    if verbose:
        print("Loading weights...")

    name_weights = params['weights']
    if not name_weights.endswith('.h5'):
        name_weights = name_weights + '.h5'

    weights = join(path_to_data, join(params['weights_path'], name_weights))
    path_predicted = join(path_to_data, params['weights_path'] + 'predictions_' + name_weights[:-3] + '/')

    unet.load_weights(weights)

    if verbose:
        print("Predicting...")

    if not os.path.exists(path_predicted):
        os.makedirs(path_predicted)
    path_tmp = join(path_to_data, 'tmp/')
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    test_images_names = []

    path_test_patches = join(path_to_data, join(params['data_path'], 'test/images/'))
    files_test = [f for f in listdir(path_test_patches) if isfile(join(path_test_patches, f))]
    path_test_label_patches = join(path_to_data, join(params['data_path'], 'test/labels/'))

    # AUC
    gt_ex = np.zeros((len(files_test), patch_size, patch_size))
    pred_ex = np.zeros((len(files_test), patch_size, patch_size))
    gt_he = np.zeros((len(files_test), patch_size, patch_size))
    pred_he = np.zeros((len(files_test), patch_size, patch_size))
    gt_ma = np.zeros((len(files_test), patch_size, patch_size))
    pred_ma = np.zeros((len(files_test), patch_size, patch_size))
    gt_se = np.zeros((len(files_test), patch_size, patch_size))
    pred_se = np.zeros((len(files_test), patch_size, patch_size))

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

        if verbose:
            print("Saving predicted image...")

        img_pred = np.zeros(pred.shape + (3,), dtype=np.uint8)
        for ind in range(len(color_code_labels)):
            img_pred[pred == ind, :] = color_code_labels[ind]

        label_pred = np.zeros(pred.shape + (3,), dtype=np.uint8)
        label_pred[:,:,0] = pred

        imsave(path_tmp + img_name + '_pred.png', img_pred)
        imsave(path_tmp + img_name + '_lab_pred.png', label_pred)

        # AUC PR
        a,b,c = file.split("_")
        label_name = a + '_' + b + '_label_' + c[:-4] + '.png'
        label = imread(path_test_label_patches + label_name)

        ex_image = np.zeros((label.shape[0], label.shape[1]))
        ex_image[label[:,:,0]==1] = 1
        gt_ex[i, :, :] = ex_image
        pred_ex[i, :, :] = pred_prob[:, :, 1]

        he_image = np.zeros((label.shape[0], label.shape[1]))
        he_image[label[:,:,0]==2] = 1
        gt_he[i, :, :] = he_image
        pred_he[i, :, :] = pred_prob[:, :, 2]

        ma_image = np.zeros((label.shape[0], label.shape[1]))
        ma_image[label[:,:,0]==3] = 1
        gt_ma[i, :, :] = ma_image
        pred_ma[i, :, :] = pred_prob[:, :, 3]

        se_image = np.zeros((label.shape[0], label.shape[1]))
        ma_image[label[:, :, 0] == 4] = 1
        gt_se[i, :, :] = se_image
        pred_se[i, :, :] = pred_prob[:, :, 4]

        i += 1

    if verbose:
        print("Prediction done")

    file = path_predicted + 'evalutation_' + params['weights'] + '.txt'
    f = open(file, "a+")
    f.write('\nAUC Precision-Recall for weights ' + params['weights'] + '\n')

    # log AUC PR
    auc_ex = AUC_PR(gt_ex, pred_ex)
    f.write("Class EX: {} \n".format(auc_ex))
    auc_he = AUC_PR(gt_he, pred_he)
    f.write("Class HE: {} \n".format(auc_he))
    auc_ma = AUC_PR(gt_ma, pred_ma)
    f.write("Class MA: {} \n".format(auc_ma))
    auc_se = AUC_PR(gt_se, pred_se)
    f.write("Class SE: {} \n".format(auc_se))

    f.close()

    # reconstruction of images from the predicted patches
    if verbose:
        print("Reconstructing predicted images...")

    path_imgs = join(path_to_data, params['images_path'])
    imgs = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]
    num_patches = compute_num_patches(join(path_imgs, imgs[0]), patch_size)
    img_aux = imread(path_imgs + imgs[0])

    for file in test_images_names:
        full_image = np.zeros((3200, 4800, 3), dtype=np.uint8)
        full_label = np.zeros((3200, 4800, 3), dtype=np.uint8)

        for i in range(int(num_patches)):
            patch_name = file + '_' + str(i+1).zfill(3) + '_pred.png'
            patch_lab_name = file + '_' + str(i + 1).zfill(3) + '_lab_pred.png'

            patch = imread(path_tmp + patch_name)
            lab_patch = imread(path_tmp + patch_lab_name)

            if i < 3:
                full_image[0:1600, i*1600:(i+1)*1600, :] = patch
                full_label[0:1600, i*1600:(i+1)*1600, :] = lab_patch
            else:
                full_image[1600:(2*1600), (i-3)*1600:(i-2)*1600, :] = patch
                full_label[1600:(2 * 1600), (i - 3) * 1600:(i - 2) * 1600, :] = lab_patch

        image_labels = full_image[:img_aux.shape[0], :img_aux.shape[1], :]
        image_labels_norm = full_label[:img_aux.shape[0], :img_aux.shape[1], :]

        imsave(path_predicted + file + '_pred.png', image_labels)
        imsave(path_predicted + file + '_lab_pred.png', image_labels_norm)

        if verbose:
            print("Predictions of {} saved".format(file))

    os.system('rm ' + path_tmp + '*')
    os.system('rmdir ' + path_tmp)


if __name__ == '__main__':
    predict(**vars(parse_arguments_pred()))