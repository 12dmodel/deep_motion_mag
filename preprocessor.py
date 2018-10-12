import tensorflow as tf
import numpy as np


def preprocess_image(image, train_config):
    pnoise = train_config["poisson_noise_n"]
    gnoise = train_config["gauss_noise_n"]

    if pnoise is not None:
        image = preproc_poisson_noise(image, pnoise)
    if gnoise is not None:
        image = preproc_gaussian_noise(image, gnoise)
    return image


def get_possion_noise(image):
    """Add poisson noise.

    This function approximate posson noise upto 2nd order.
    Assume images were in 0-255, and converted to the range of -1 to 1.
    """
    n = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0)
    # strength ~ sqrt image value in 255, divided by 127.5 to convert
    # back to -1, 1 range.
    n_str = tf.sqrt(image + 1.0) / np.sqrt(127.5)
    return tf.multiply(n, n_str)


def get_gaussian_noise(image):
    return tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0)


def preproc_poisson_noise(image, n):
    nn = np.random.uniform(0, n)
    return image + nn * get_possion_noise(image)


def preproc_gaussian_noise(image, n):
    # Scale N so that it is meaningful in 0-255 scale.
    nn = np.random.uniform(0, n)
    nn = nn / 127.5
    return image + nn * get_possion_noise(image)


def preproc_color(image, n):
    n = tf.to_float(n) / 127.5 - 1
    image += n
    return tf.clip_by_value(image, -1.0, 1.0)
