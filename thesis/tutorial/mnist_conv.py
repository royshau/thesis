# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
import tensorflow as tf
from tutorial import ops
IM_W = 28
IM_H = 28


class DeepMnist(object):
    """ This class represents convolutional network for mnist """
    def __init__(self, input_, labels, FLAGS):
        """
        Constructor
        """
        self.FLAGS = FLAGS
        self.input = input_
        self.labels = labels
        self.loss = None
        self.predict = None
        self.train_step = None
        self.evaluation = None

    def build(self):
        """ build model """
        with tf.name_scope('architecture'):
            self.predict = self.architecture()

        with tf.name_scope('loss'):
            self.loss = self.calc_loss()

        with tf.name_scope('train_step'):
            self.train_step = self.solver()

        with tf.name_scope('evaluation'):
            self.evaluation = self.calc_evaluation()

    def architecture(self):
        """From tensorflow website
            deepnn builds the graph for a deep net for classifying digits.
            Args:
                x: an input tensor with the dimensions (N_examples, 784), where 784 is the
                number of pixels in a standard MNIST image.
            Returns:
                A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
                equal to the logits of classifying the digit into one of 10 classes (the
                digits 0-9). keep_prob is a scalar placeholder for the probability of
                dropout.
        """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.input, [-1, IM_W, IM_H, 1])

        # Dump output tensoboard
        tf.summary.image(name='Input image', tensor=x_image, max_outputs=5)

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        conv1 = ops.conv_2d(input_=x_image, k_h=5, k_w=5, n_filters=32, d_h=1, d_w=1, hist=True, name='conv1')
        relu1 = tf.nn.relu(conv1, name='relu1')
        pool1 = ops.max_pool_2x2(relu1, name='pool1')   # Pooling layer - downsamples by 2X.

        # Second convolutional layer -- maps 32 feature maps to 64.
        conv2 = ops.conv_2d(input_=pool1, k_h=5, k_w=5, n_filters=64, d_h=1, d_w=1, hist=True, name='conv2')
        relu2 = tf.nn.relu(conv2, name='relu2')
        pool2 = ops.max_pool_2x2(relu2, name='pool2')   # Pooling layer - downsamples by 2X.

        with tf.name_scope('flatten'):
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        fc1 = ops.fully_connected(input_=pool2_flat, output_size=1024, hist=True, name='fc1')
        fc1_relu = tf.nn.relu(fc1, name='fc1_relu')

        # Map the 1024 features to 10 classes, one for each digit
        predict = ops.fully_connected(input_=fc1_relu, output_size=10, hist=True, name='predict')

        # Dump output tensoboard
        tf.summary.image(name='output', tensor=tf.reshape(predict, shape=(-1, 10, 1, 1)), max_outputs=5)

        return predict

    def calc_loss(self):
        """ Define loss function"""
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predict)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # Summary tensorboard
        tf.summary.scalar('loss', cross_entropy_mean)

        return cross_entropy_mean

    def solver(self):
        """ Define solver """
        solver_adam = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        train_step = solver_adam.minimize(self.loss)
        return train_step

    def calc_evaluation(self):
        """ Calculate performance"""
        correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        # Summary tensorboard
        tf.summary.scalar('accuracy', accuracy)
        return accuracy