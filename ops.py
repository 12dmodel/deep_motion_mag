import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from utils import *


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth],
                                initializer=
                                tf.random_normal_initializer(1.0, 0.02,
                                                             dtype=tf.float32))
        offset = tf.get_variable("offset", [depth],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02,
           padding='SAME', name="conv2d", activation_fn=None,
           weights_regularizer=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                           activation_fn=activation_fn,
                           weights_initializer=
                           tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None,
                           weights_regularizer=weights_regularizer)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02,
             name="deconv2d", activation_fn=None, weights_regularizer=None):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME',
                                     activation_fn=activation_fn,
                                     weights_initializer=
                                     tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None,
                                     weights_regularizer=weights_regularizer)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02,
           bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix",
                                 [input_.get_shape()[-1], output_size],
                                 tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start)
                              )
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def expand_dims_1_to_4(tensor, dims=None):
    """Expand dimension from 1 to 4.

    Useful for multiplying amplification factor.
    """
    if not dims:
        dims = [-1, -1, -1]
    return tf.expand_dims(
             tf.expand_dims(
               tf.expand_dims(tensor, dims[0]),
               dims[1]),
             dims[2])


def residual_block(x, output_dim, ks=3, s=1, name='residual_block'):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = conv2d(y, output_dim, ks, s, padding='VALID', name=name+'_c1')
    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = conv2d(y, output_dim, ks, s, padding='VALID', name=name+'_c2')
    return y + x
