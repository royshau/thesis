# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/ops.py
#   + License: MIT

import tensorflow as tf
from tensorflow.python.framework import ops


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    # with ops.op_scope([preds, targets], name, "bce_loss") as name:
    with tf.name_scope(name=name,  default_name="bce_loss", values=[preds, targets]) as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, in_channels=None, data_format='NCHW', name="conv2d", hist=False):

    if in_channels is None:
        in_channels = input_.get_shape()[-1] if data_format == 'NHWC' else input_.get_shape()[1]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, in_channels, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, d_h, d_w], padding='SAME', data_format=data_format)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases, data_format=data_format)
        collect_reg(dict(w=w, b=biases))

        if hist:
            tf.summary.histogram(name=name+"_w", values=w, collections='G')
            tf.summary.histogram(name=name + "_b", values=biases, collections='G')
        return conv


def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d_transpose", data_format='NCHW', with_w=False):
    in_channels = input_.get_shape()[-1] if data_format == 'NHWC' else input_.get_shape()[1]
    out_channels = output_shape[-1] if data_format == 'NHWC' else output_shape[1]
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, out_channels, in_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        try:
            # TODO: Currently support only NCHW
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, 1, d_h, d_w], data_format=data_format)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, 1, d_h, d_w], data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases, data_format=data_format)

        collect_reg(dict(w=w, b=biases))
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False, hist=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if hist:
            tf.summary.histogram(name="linear" + "_w", values=matrix, collections='G')
            tf.summary.histogram(name="linear" + "_b", values=bias, collections='G')

        collect_reg(dict(w=matrix, b=bias))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def batch_norm(in_tensor, phase_train, name, decay=0.99, data_format='NCHW'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
        decay:       decay factor
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name) as scope:
        return tf.contrib.layers.batch_norm(in_tensor, is_training=phase_train, decay=decay, scope=scope, fused=True,
                                            data_format=data_format)


def res_block(input_, output_dim, train_phase, k_h=5, k_w=5, d_h=2, d_w=2, in_channels=None, data_format='NCHW', name="conv2d"):
    """
    Define residual block
    :param input_:
    :param output_dim:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param in_channels:
    :param train_phase:
    :param name:
    :return:
    """

    conv_1 = conv2d(input_, output_dim=output_dim, k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                               in_channels=in_channels, data_format=data_format, name=name+"_conv_1")
    conv_1_bn = batch_norm(conv_1, train_phase, decay=0.98, name=name+"_bn_1", data_format=data_format)
    relu_1 = tf.nn.relu(conv_1_bn, name=name+"_relu_1")

    conv_2= conv2d(relu_1, output_dim=output_dim, k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                               in_channels=in_channels, data_format=data_format, name=name+"_conv_2")
    conv_2_bn = batch_norm(conv_2, train_phase, decay=0.98, name=name+"_bn_2", data_format=data_format)

    addition = input_ + conv_2_bn
    relu_2 = tf.nn.relu(addition, name=name+"_relu_2")

    return relu_2

def collect_reg(reg_dict):
    """
    Collect all regularization weights
    :param reg_dict:
    :return:
    """
    for (reg_name, var) in reg_dict.iteritems():
        # Currently only L2
        reg_value = tf.nn.l2_loss(var, name='reg_' + reg_name)
        tf.add_to_collection("regularization_" + reg_name, reg_value)