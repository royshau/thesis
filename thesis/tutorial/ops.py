import tensorflow as tf


def conv_2d(input_, k_h, k_w, n_filters, d_h=1, d_w=1, hist=False, name='conv'):
    """ Conv layer wrapper
    :param input_:  input layer
    :param k_h: kernel height
    :param k_w: kernel width
    :param n_filters: number of output filters
    :param d_h: stride height
    :param d_w: stride width
    :param hist: dump histograms
    :param name: name of scope
    :return:
    """
    with tf.variable_scope(name):
        in_channels = input_.get_shape()[-1]
        weights = tf.get_variable('w', [k_h, k_w, in_channels, n_filters],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('b', [n_filters], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = conv + biases

        if hist:
            tf.summary.histogram(name=name + "_w", values=weights)
            tf.summary.histogram(name=name + "_b", values=biases)
        return conv


def fully_connected(input_, output_size, hist=False, name='fc'):
    """ Fully connected layer
    :param input_: input layer
    :param output_size: output channels size
    :param hist: dump histograms
    :param name: name of scope
    :return:
    """
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        weights = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        if hist:
            tf.summary.histogram(name="fc" + "_w", values=weights)
            tf.summary.histogram(name="fc" + "_b", values=biases)

        return tf.matmul(input_, weights) + biases


def max_pool_2x2(x, name='pool'):
    """max_pool_2x2 downsamples a feature map by 2X."""
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
