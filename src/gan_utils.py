from keras.preprocessing.image import Iterator
import numpy as np
from os import listdir
from os.path import isfile, join

from skimage.io import imread
from keras.utils import to_categorical
from random import shuffle, randint
import time

from data import image_shape, compute_statistics_file

train_path = '/imatge/ngullon/work/retina_data/train/'
labels = ['EX/', 'HE/', 'MA/', 'SE/']
data_path = '/imatge/ngullon/work/retina_data/'

channels = 3
n_classes = 5
out_channels = 4


class TrainBatchFetcher(Iterator):
    """
    Fetch batch of original images and labels to train/evaluate the network
    """
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.n_train_imgs = self.images.shape[0]
        self.batch_size = batch_size

    def next(self):
        indices = list(np.random.choice(self.n_train_imgs, self.batch_size, replace=False))
        return self.images[indices, :, :, :], self.labels[indices, :, :, :]


def get_data(path, thres_score = None):
    """
    It gets the image and labels (with more ratio of lesion than the given threshold 'thres_score') of the path passed as a parameter
    :param path: path where images and labels are located. Images should be inside a folder called 'images' and labels inside a folder called 'labels'
    :param thres_score: threshold of lesion/pixels
    :return: the images, the labels and the number of images returned
    """
    files = [f for f in listdir(path + 'images/') if isfile(join(path + 'images/', f))]

    if thres_score is not None:
        files_aux = []
        for file_name in files:
            score = compute_statistics_file(path, file_name)
            if score > thres_score:
                files_aux.append(file_name)
        files = files_aux
    shuffle(files)
    img_shape = image_shape(path + 'images/' + files[0])

    images_train = np.zeros((len(files), img_shape[0], img_shape[1], channels), dtype=np.uint8)
    labels_train = np.zeros((len(files), img_shape[0], img_shape[1], n_classes), dtype=np.uint8)

    n = 0
    for file_name in files:
        img = imread(path + 'images/' + file_name)
        image_array = np.asarray(img).astype(np.uint8)

        a, b, c = file_name.split("_")

        label_name = a + '_' + b + '_label_' + c[:-4] + '.tif'
        label_img = imread(path + 'labels/' + label_name)
        label_array = np.asarray(label_img[:, :, 0]).astype(np.uint8)
        assert (np.amin(label_img) >= 0 and np.amax(label_img) <= 5)

        images_train[n, :, :, :] = image_array
        labels_train[n, :, :, :] = to_categorical(label_array, n_classes).reshape(label_array.shape + (n_classes,))

        n += 1
        if (n+1)%100 == 0 or (n+1) == len(files):
            print "Done: {0}/{1}".format((n+1), len(files))

    return images_train, labels_train, len(files)


def imgs2discr(real_images, real_labels, fake_labels):
    """
    It gets the input data to the discriminator
    :param real_images: input images
    :param real_labels: ground truth
    :param fake_labels: predicted labels
    :return: input images and labels to the discriminative network
    """
    real = np.concatenate((real_images, real_labels), axis=3)
    fake = np.concatenate((real_images, fake_labels), axis=3)

    img_batch = np.concatenate((real, fake), axis=0)
    lab_batch = np.ones((img_batch.shape[0], 1))
    lab_batch[real.shape[0]:,...] = 0

    return img_batch, lab_batch


def imgs2gan(real_images, real_labels):
    """
    It gets the input data to the segmentation network
    :param real_images: input images
    :param real_labels: ground truth
    :return: input images and labels to the segmentation network
    """
    img_batch = [real_images, real_labels]
    lab_batch = np.ones((real_images.shape[0], 1))

    return img_batch, lab_batch


if __name__ == '__main__':
    path = '/imatge/ngullon/work/retina_data/400_train/'
    img, lab = get_data(path)
    a = TrainBatchFetcher(images=img, labels=lab, batch_size=5)