"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import numpy as np
import copy
import os
import errno
import cv2

pp = pprint.PrettyPrinter()


# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_train_data(image_path, gray_scale=True, is_testing=False):
    img_A = imread(image_path[0], gray_scale)
    img_B = imread(image_path[1], gray_scale)
    img_Out = imread(image_path[2], gray_scale)
    if not is_testing:
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            img_Out = np.fliplr(img_Out)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_Out = img_Out/127.5 - 1.
    img_A, img_B, img_Out = np.atleast_3d(img_A, img_B, img_Out)
    img_AB_out = np.concatenate((img_A, img_B, img_Out), axis=2)
    return img_AB_out


def get_image(image_path,
              image_size,
              is_crop=True,
              resize_w=64,
              is_grayscale=False):
    return transform(imread(image_path, is_grayscale),
                     image_size,
                     is_crop,
                     resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=True):
    if (is_grayscale):
        return cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float)
    else:
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, size, path):
    im = merge(images, size)
    if issubclass(im.dtype.type, np.floating):
        im = im * 255
        im = im.astype('uint8')
    return cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(d):
            pass
        else:
            raise
