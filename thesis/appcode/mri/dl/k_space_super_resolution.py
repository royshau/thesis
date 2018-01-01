import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolution(BasicModel):
    """
    Represents k-space super resolution model for multi-channel input
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=input, labels=labels, dims_in=dims_in, dims_out=dims_out)
        self.x_input_upscale = None

    def model(self):
        """
        Define the model
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(self.input, [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]],
                             name='x_input_reshaped')

        # Apply image resize
        x_image_upscale = tf.image.resize_bilinear(x_image, np.array([self.dims_out[0],
                                          self.dims_out[1]]), align_corners=None, name='x_input_upscale')

        self.x_input_upscale = x_image_upscale
        # Dump input image out
        tf.image_summary('x_upscale', x_image_upscale)

        # Model convolutions
        conv_1 = ops.conv2d(x_image_upscale, output_dim=8, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_1")
        relu_1 = tf.nn.relu(conv_1)

        conv_2 = ops.conv2d(relu_1, output_dim=4, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_2")
        relu_2 = tf.nn.relu(conv_2)

        conv_3 = ops.conv2d(relu_2, output_dim=1, k_h=1, k_w=1, d_h=1, d_w=1, name="conv_3")
        relu_3 = tf.nn.relu(conv_3)

        conv_4 = ops.conv2d(relu_3, output_dim=1, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_4")

        predict = tf.reshape(conv_4, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='predict')

        # Dump prediction out
        tf.image_summary('predict', predict)
        return predict

    def loss(self, predict, labels):
        """
        Return loss value
        :param predict: prediction from the model
        :param labels: labels input
        :return:
        """
        s_loss = tf.reduce_mean(tf.square(tf.squeeze(predict) - labels), name='Loss')
        _ = tf.scalar_summary('square-loss', s_loss)
        return s_loss

    def training(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(s_loss.op.name, s_loss)
        # Create Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(s_loss, global_step=global_step)
        return train_op

    def evaluation(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """
        evalu = tf.reduce_mean(tf.square(tf.squeeze(predict) - labels))
        tf.scalar_summary("Evaluation", evalu)
        # Return the number of true entries.
        return evalu
