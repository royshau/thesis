#!/home/ohadsh/Tools/anaconda/bin/python
""" Train to complete k_space lines with factor 2
k_space real + imaginary are scaled with log function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.deep_learning.helpers import *

# k space data set
base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
file_names = ['k_space_real', 'k_space_real_gt']

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('mini_batch_size', 20, 'Size of mini batch')
flags.DEFINE_integer('print_test', 10000, 'Print test frequency')
flags.DEFINE_boolean('to_show', False, 'View data')
tf.app.flags.DEFINE_string('train_dir', '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Init inputs as patheholders
x = tf.placeholder(tf.float32, shape=[None, 128, 256], name='x_input')
y_ = tf.placeholder(tf.float32, shape=[None, 256, 256], name='y_input')

# Reshape the input for batchSize, 28x28 image, 1 channel
x_image = tf.reshape(x, [-1, 128, 256, 1], name='x_input_reshape')
y_image = tf.reshape(y_, [-1, 256, 256, 1], name='y_input_reshape')

x_image_upscaled = tf.image.resize_bilinear(x_image,
                                            np.array([256, 256]), align_corners=None, name='x_input_upscaled')

# Init all W and b:
# First convolutional layer weights
W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])

# Second convolutional layer weights
W_conv2 = weight_variable([1, 1, 8, 1])
b_conv2 = bias_variable([1])

# Third convolutional layer weights - reconstruction
W_conv3 = weight_variable([1, 1, 1, 1])
b_conv3 = bias_variable([1])

# Network
# First conv & relu
h_conv1 = tf.nn.relu(conv2d(x_image_upscaled, W_conv1) + b_conv1)

# Second conv & relu
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# Thirs conv
h_conv3 = conv2d(h_conv2, W_conv3) + b_conv3

y_pred = tf.reshape(h_conv3, [-1, 256, 256, 1], name='y_pred')

image_summary = tf.image_summary('x_upscaled', x_image_upscaled)
image_summary = tf.image_summary('y_pred', y_pred)

# More name scopes will clean up the graph representation
with tf.name_scope('xent'):
    # Loss
    square_loss = tf.reduce_mean(tf.square(y_image - y_pred))
    _ = tf.scalar_summary('square-loss', square_loss)

with tf.name_scope('train'):
    # Training evaluation
    # Using Adam solver with cross entropy minimize
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(square_loss)

with tf.name_scope('test'):
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.square(y_image - y_pred))
    _ = tf.scalar_summary('accuracy', accuracy)