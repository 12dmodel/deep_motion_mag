from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import numpy as np


def res_manipulator(enc_a,
                    enc_b,
                    amplification_factor,
                    layer_dims,
                    num_resblk,
                    num_conv,
                    num_aft_conv=0,
                    probe_pt=None):
    diff = (enc_b - enc_a)
    if probe_pt is not None:
        probe_pt["mani_diff"] = diff
    for i in range(num_conv):
        p = 3
        k = 7
        diff = tf.pad(diff, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        cname = 'mani_conv{}'.format(i)
        diff = tf.nn.relu(conv2d(diff, layer_dims, k, 1, padding='VALID', name=cname + 'c'))
    if probe_pt is not None:
        probe_pt["mani_after_conv"] = diff
    diff = diff * expand_dims_1_to_4(amplification_factor - 1.0)
    if probe_pt is not None:
        probe_pt["mani_after_mult"] = diff
    for i in range(num_aft_conv):
        diff = tf.pad(diff, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        cname = 'mani_aft_conv{}'.format(i)
        diff = conv2d(diff, layer_dims, 3, 1, padding='VALID', name=cname + 'c')
    for i in range(num_resblk):
        diff = residual_block(diff, layer_dims, 3, 1, name='mani_resblk{}'.format(i))
    if probe_pt is not None:
        probe_pt["mani_after_res"] = diff
    return enc_b + diff


def res_encoder(image, layer_dims, num_resblk):
    # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
    # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
    # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
    c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    c1 = tf.nn.relu(conv2d(c0, layer_dims / 2, 7, 1, padding='VALID', name='enc_conv1_c'))
    c2 = tf.nn.relu(conv2d(c1, layer_dims, 3, 2, name='enc_conv2_c'))
    # define G network with 9 resnet blocks
    r = c2
    for i in range(num_resblk):
        r = residual_block(r, layer_dims, 3, 1, name='encoder_resblk{}'.format(i))
    return r


def res_decoder(activation,
                layer_dims,
                out_channels,
                num_resblk):
    r = activation
    for i in range(num_resblk):
        r = residual_block(r, layer_dims, 3, 1, name='decoder_resblk{}'.format(i))
    up = tf.image.resize_nearest_neighbor(r, tf.shape(r)[1:3] * 2)
    up = tf.pad(up, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    d2 = tf.nn.relu(conv2d(up, int(layer_dims / 2), 3, 1, padding='VALID', name='dec_conv2_c'))
    d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    out = conv2d(d2, out_channels, 7, 1, padding='VALID', name='pred_conv')
    return out


def L1_loss(in_, target):
    with tf.variable_scope("l1_loss"):
        return tf.reduce_mean(tf.abs(in_ - target))
