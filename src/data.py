from os import listdir
from os.path import isfile, join
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage.util.shape import view_as_windows
from skimage.morphology import opening, closing, dilation
from random import shuffle
import random
import os
from utils.params import parse_arguments, default_params



path_to_data = '../'

def create_datasets(**params):
    """
    It creates train, validation and test datasets with the given ratios
    It creates masks for all datasets
    For train and validation datasets, it also creates patches of size patch_size (in order to fit in memory when training) for the masked images and labels (all classes in one label with values from 0 to 4)
    For test dataset, it creates patches of size patch_size_test (in order to fit in memory when predicting) for the masked images
    :param files_path: path where images are located
    :param data_path: path where datasets will be created. Inside this directory, there is a folder for each class with their labels inside
    :param ratio_test: ratio to do the partition for test dataset
    :param ratio_val: ratio to do the partition for validation dataset
    :param patch_size: patch size of train and validation images
    :param patch_size_test: patch size of test images
    :param channels: number of channels of images
    """

    params = dict(
        default_params,
        **params
    )

    verbose = params['verbose']

    data_path = join(path_to_data, params['data_path'])
    train_path = join(data_path, 'train/')
    test_path = join(data_path, 'test/')
    val_path = join(data_path, 'val/')
    path_mask = join(data_path, 'masks/')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    rand_seed = params['seed']
    random.seed(rand_seed)

    if verbose:
        print("Preprocessing and preparing data...")

    files_path = join(path_to_data, params['images_path'])

    files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    if verbose:
        print("Files used: {}".format(len(files)))

    # Very few samples have SE class, so we make sure images test and validation sets have images with SE
    path_se_class = join(path_to_data, join(params['gt_path'], 'SE/'))
    files_se = [f for f in listdir(path_se_class) if isfile(join(path_se_class, f))]

    n_test = int(np.round(len(files) * params['ratio_test']))
    n_val = int(np.round(len(files) * params['ratio_val']))

    files_test = []
    files_val = []
    files_train = []

    shuffle(files_se)
    shuffle(files)
    for i in range(len(files_se)):
        name_file = files_se[i][:-7] + '.jpg'
        if i < n_test:
            files_test.append(name_file)
        elif i < n_test + n_val:
            files_val.append(name_file)
        else:
            files_train.append(name_file)
    for file in files:
        if file not in files_train and file not in files_test and file not in files_val:
            files_train.append(file)

    if verbose:
        print("Files in train: {}".format(len(files_train)))
        print("Files in validation: {}".format(len(files_val)))
        print("Files in test: {}".format(len(files_test)))

    patch_size = params['patch_size']
    patch_size_test =  params['patch_size_test']
    channels = params['channels']
    threshold_mask = params['thres_mask']
    labels = params['labels']
    gt_path = join(path_to_data, params['gt_path'])

    img_aux = imread(files_path + files[0])
    img_size = img_aux.shape
    pad_height = 0
    pad_width = 0
    if img_size[0] % patch_size != 0:
        num = int(img_size[0] / patch_size) + 1
        pad_height = int(patch_size * num - img_size[0])
    if img_size[1] % patch_size != 0:
        num = int(img_size[1] / patch_size) + 1
        pad_width = int(patch_size * num - img_size[1])

    count = 0
    for file in files_train:
        img_name = file[:-4]
        image = imread(files_path + file)

        mask_name = img_name + '_mask'
        create_data_mask(image, mask_name, path_mask, threshold=threshold_mask)
        mask = imread(path_mask + mask_name + '.png') // 255
        masked_img = mask * image
        image_pad = np.pad(masked_img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        path_patches_train = train_path + 'images/'
        create_patches(image_pad, img_name, patch_size, patch_size // 2, channels, path_patches_train, '.png')

        image_all_labels = np.zeros((img_size[0] + pad_height, img_size[1] + pad_width, channels), dtype=np.uint8)
        for lab in range(len(labels)):
            try:
                image = imread(gt_path + labels[lab] + img_name + '_' + labels[lab][:-1] + '.tif')
                image_pad = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                label = np.asarray(image_pad).astype(np.uint8)
            except IOError:
                label = np.zeros((img_size[0] + pad_height, img_size[1] + pad_width, channels), dtype=np.uint8)
            image_all_labels[(label[:, :, 0] == 255)] = lab+1
        path_patches_train_all_labels = train_path + 'labels/'
        create_patches(image_all_labels, img_name + '_label', patch_size, patch_size // 2, channels, path_patches_train_all_labels, '.png')

        if verbose:
            count += 1
            print("Done: {0}/{1} of train dataset".format(count, len(files_train)))

    if verbose:
        print("Patches created for train dataset")

    count = 0
    for file in files_val:
        img_name = file[:-4]
        image = imread(files_path + file)

        mask_name = img_name + '_mask'
        create_data_mask(image, mask_name, path_mask, threshold=threshold_mask)
        mask = imread(path_mask + mask_name + '.png') // 255
        masked_img = mask * image
        image_pad = np.pad(masked_img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        path_patches_val = val_path + 'images/'
        create_patches(image_pad, img_name, patch_size, patch_size // 2, channels, path_patches_val, '.png')

        image_all_labels = np.zeros((img_size[0] + pad_height, img_size[1] + pad_width, channels), dtype=np.uint8)
        for lab in range(len(labels)):
            try:
                image = imread(gt_path + labels[lab] + img_name + '_' + labels[lab][:-1] + '.tif')
                image_pad = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                label = np.asarray(image_pad).astype(np.uint8)
            except IOError:
                label = np.zeros((img_size[0] + pad_height, img_size[1] + pad_width, channels), dtype=np.uint8)
            image_all_labels[(label[:, :, 0] == 255)] = lab+1
        path_patches_val_all_labels = val_path + 'labels/'
        create_patches(image_all_labels, img_name + '_label', patch_size, patch_size // 2, channels, path_patches_val_all_labels, '.png')

        if verbose:
            count += 1
            print("Done: {0}/{1} of validation dataset".format(count, len(files_val)))

    if verbose:
        print("Patches created for validation dataset")

    pad_height = 0
    pad_width = 0
    if img_size[0] % patch_size != 0:
        num = int(img_size[0] / patch_size_test) + 1
        pad_height = int(patch_size_test * num - img_size[0])
    if img_size[1] % patch_size_test != 0:
        num = int(img_size[1] / patch_size_test) + 1
        pad_width = int(patch_size_test * num - img_size[1])

    count = 0
    for file in files_test:
        img_name = file[:-4]
        image = imread(files_path + file)

        mask_name = img_name + '_mask'
        create_data_mask(image, mask_name, path_mask, threshold=threshold_mask)
        mask = imread(path_mask + mask_name + '.png') // 255
        masked_img = mask * image
        image_pad = np.pad(masked_img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        path_test_images = test_path + 'full_images/'
        if not os.path.exists(path_test_images):
            os.makedirs(path_test_images)
        imsave(path_test_images + img_name + '.jpg', masked_img)
        path_patches_test = test_path + 'images/'
        create_patches(image_pad, img_name, patch_size_test, patch_size_test, channels, path_patches_test, '.png')

        image_all_labels = np.zeros((img_size[0], img_size[1], channels), dtype=np.uint8)
        for lab in range(len(labels)):
            try:
                image = imread(gt_path + labels[lab] + img_name + '_' + labels[lab][:-1] + '.tif')
                label = np.asarray(image).astype(np.uint8)
            except IOError:
                label = np.zeros((img_size[0], img_size[1], channels), dtype=np.uint8)
            image_all_labels[(label[:, :, 0] == 255)] = lab + 1
        path_test_all_full_labels = test_path + 'full_labels/'
        if not os.path.exists(path_test_all_full_labels):
            os.makedirs(path_test_all_full_labels)
        imsave(path_test_all_full_labels + img_name + '_label.png', image_all_labels)
        image_all_labels_pad = np.pad(image_all_labels, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        path_patches_test_all_labels = test_path + 'labels/'
        create_patches(image_all_labels_pad, img_name + '_label', patch_size_test, patch_size_test, channels,
                       path_patches_test_all_labels, '.png')

        if verbose:
            count += 1
            print("Done: {0}/{1} of test dataset".format(count, len(files_test)))

    if verbose:
        print("Patches created for test dataset")


def image_shape(filename):
    """
    Reads and computes the size of the file passed as a parameter
    :param filename: path and name of the file
    :return: array with the size of the file
    """
    img = imread(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def create_data_mask(image, img_name, out_path, threshold):
    """
    It creates a mask of the image array passed as a parameter and save the mask created in the out_path with the img_name
    :param image: image array
    :param img_name: name of the image to be saved
    :param out_path: path where the mask created is saved
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mask_index = (image < threshold)

    mask = (~mask_index).astype(np.uint8) * 255
    str_element_op = np.ones((11, 11, 3))
    str_element_cl = np.ones((11, 11, 3))
    str_element_dil = np.ones((21, 21, 3))
    mask_op = opening(mask, str_element_op)
    mask_cl = closing(mask_op, str_element_cl)
    mask_dil = dilation(mask_cl, str_element_dil)

    imsave(out_path + img_name + '.png', mask_dil)


def create_patches(image, img_name, patch_size, step_size, channels, out_path, out_format):
    """
    It creates patches of size patch_size of the image array passed as a parameter with a step of step_size (overlap)
    The patches generated are saved in the out_path with the img_name and in out_format
    :param image: image array
    :param img_name: name of the image to be saved
    :param patch_size: size of the patches
    :param step_size: size of the overlapping
    :param channels: channels of the patches to be created
    :param out_path: path in which the patches will be saved
    :param out_format: format in which the patches will be saved
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    window_shape = (patch_size, patch_size, channels)
    patches_image = view_as_windows(image, window_shape, step=step_size)
    i = 1
    for group_patches in patches_image:
        for patch in group_patches:
            imsave(out_path + img_name + '_' + str(i).zfill(3) + out_format, patch[0])
            i += 1


def compute_statistics_file(label_img):
    """
    It computes the ratio between the number of pixels belonging to lesion and all the pixels of the patch
    :param label_img: label array with all the classes
    """
    label_array = np.asarray(label_img).astype(np.uint8)
    lab = label_array[:,:,0]
    lab_size = lab.shape
    n_pixels = float(lab_size[0] * lab_size[1])
    lab_score = np.sum(lab != 0) / n_pixels

    return lab_score


def create_labels_color(**params):
    """
    It creates an image with all the labels in the same image. Each label is printed with a different color following the color_code_labels
    :param files_path: path where images are located
    :param data_path: path where labels (segmentation maps) are located. Inside this directory, there should a folder for each class,
            where the label for that class is located
    :param all_labels_path: path where the created images are saved
    """

    params = dict(
        default_params,
        **params
    )

    verbose = params['verbose']

    files_path = join(path_to_data, params['images_path'])
    all_labels_path = join(path_to_data, join(params['gt_path'], 'all_labels/'))
    gt_path = join(path_to_data, params['gt_path'])
    labels = params['labels']

    if not os.path.exists(all_labels_path):
        os.makedirs(all_labels_path)

    files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    img_aux = imread(files_path + files[0])
    img_size = img_aux.shape

    color_code_labels = params['color_code_labels']

    if verbose:
        print("Creating ground truth with all the classes in the same label...")

    for file in files:
        img_name = file[:-4]
        image_all_labels = np.zeros((img_size[0], img_size[1], params['channels']), dtype=np.uint8)
        for lab in range(len(labels)):
            try:
                image = imread(gt_path + labels[lab] + img_name + '_' + labels[lab][:-1] + '.tif')
                label = np.asarray(image).astype(np.uint8)
            except IOError:
                label = np.zeros((img_size[0], img_size[1], img_size[2]), dtype=np.uint8)
            image_all_labels[(label[:,:,0]==255)] = color_code_labels[lab+1]
        imsave(all_labels_path + img_name + '_all_labels.png', image_all_labels)

    if verbose:
        print("Done!")


def compute_mean_and_std(files_path, out_path):
    """
    It computes the a mean image and a standard deviation image at a pixel level from all the images in the files_path directory
    :param files_path: files where images are located
    :param out_path: files where mean and std images are saved
    """

    files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    img_aux = imread(files_path + files[0])
    img_size = img_aux.shape

    all_images = np.zeros((len(files), img_size[0], img_size[1], img_size[2]))

    i = 0
    for file in files:
        image = imread(files_path + file)
        image_array = np.asarray(image).astype(np.uint8)
        all_images[i, :, :, :] = image_array

    mean_image = np.mean(all_images, axis=0)
    std_image = np.std(all_images, axis=0)

    print("Mean and std images calculated")

    imsave(out_path + 'mean_image.tif', mean_image)
    imsave(out_path + 'std_image.tif', std_image)


def compute_num_patches(file, patch_size):
    img = imread(file)
    num_patches = np.ceil(img.shape[0] / patch_size) * np.ceil(img.shape[1] / patch_size)
    return num_patches


if __name__ == '__main__':
    create_datasets(**vars(parse_arguments()))
    create_labels_color(**vars(parse_arguments()))
