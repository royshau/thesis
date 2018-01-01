# Helpers
"""
This file contains helpers for Deep learning networks
Based on tensorflow lib
"""

import tensorflow as tf
import numpy as np


def weight_variable(shape):
    """
    Create Weight variable for tf, with normal distribution initialization
    :param shape: for example - [5, 5, 32, 64]
    :return: tf Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Create Bias variable for tf, with constant 0.1 value
    :param shape: for example [1024]
    :return: tf Variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    Create convolution layer 2d ops with stride of 1 and padding
    :param x: input
    :param W: weights
    :return: tf
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    Create polling layer ops with of 2x2 with stride 1 and padding
    :param x: input
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')