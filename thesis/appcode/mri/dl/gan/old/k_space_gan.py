import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolutionGAN(BasicModel):
    """
    Represents k-space super resolution model
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, batch_size=None,
                 reg_w=0.0, train_phase=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=input, labels=labels, dims_in=dims_in, dims_out=dims_out)
        # HACK
        self.input_d = labels
        # HACK

        self.train_phase = train_phase
        self.predict_g = None

        self.predict_d = None
        self.predict_d_logits = None

        self.predict_d_for_g = None
        self.predict_d_logits_for_g = None

        self.train_op_g = None
        self.train_op_d = None

        self.evaluation = None
        self.update_ops = None
        self.x_input_upscale = None
        self.debug = tf.shape(input)
        self.batch_size = batch_size
        self.reg_w = reg_w
        self.regularization_values = []
        self.regularization_sum = None

        self.regularization_values_d = []
        self.regularization_sum_d = None

        tf.get_collection('D')
        tf.get_collection('G')

    def build(self, FLAGS):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g = self.__G__()

        with tf.name_scope('D_'):
            self.predict_d, self.predict_d_logits = self.__D__(self.input_d, input_type="Real")
            tf.get_variable_scope().reuse_variables()
            self.predict_d_for_g, self.predict_d_logits_for_g = self.__D__(self.predict_g, input_type="Gen")

            if len(self.regularization_values_d) > 0:
                self.regularization_sum_d = sum(self.regularization_values_d)

        with tf.name_scope('loss'):
            # self.loss_g = self.__loss_g__(predict=self.predict_g, self.labels, reg=self.regularization_sum)
            self.__loss__(FLAGS)

        with tf.name_scope('training'):
            self.train_op_d, self.train_op_g = self.__training__(learning_rate=FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy L2 norm
            self.evaluation = self.__evaluation__(predict=self.predict_g, labels=self.labels)

    def __G__(self):
        """
        Define the model
        """
        # Apply image resize for debugging
        self.x_input_upscale = tf.image.resize_bilinear(self.input, np.array([self.dims_out[0],
                                          self.dims_out[1]]), align_corners=None, name='G_x_input_upscale')

        # Dump input image out
        input_real = tf.slice(self.input, begin=[0,0,0,0], size=[1,-1,-1,1], name='G_Slice_real_input')
        input_imag = tf.slice(self.input, begin=[0,0,0,1], size=[1,-1,-1,1], name='G_Slice_imag_input')

        tf.summary.image('G_x_input_real', input_real, collections='G')
        tf.summary.image('G_x_input_imag', input_imag, collections='G')

        # Model convolutions
        out_dim = 8
        self.conv_1, reg_1 = ops.conv2d(self.input, output_dim=out_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="G_conv_1")
        self.conv_1_bn = ops.batch_norm(self.conv_1, self.train_phase, decay=0.98, name="G_bn1")
        self.relu_1 = tf.nn.relu(self.conv_1_bn)
        self.regularization_values.append(reg_1)

        # deconv for get bigger image
        out_shape = [self.batch_size, self.dims_out[0], self.dims_out[1], 4]
        self.conv_2, reg_2 = ops.conv2d_transpose(self.relu_1, output_shape=out_shape,
                                                   k_h=3, k_w=3, d_h=2, d_w=1, name="G_conv_2")
        self.conv_2_bn = ops.batch_norm(self.conv_2, self.train_phase, decay=0.98, name="G_bn2")
        self.relu_2 = tf.nn.relu(self.conv_2_bn)
        self.regularization_values.append(reg_2)

        out_dim = 2
        self.conv_3, reg_3 = ops.conv2d(self.relu_2, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_conv_3")
        self.conv_3_bn = ops.batch_norm(self.conv_3, self.train_phase, decay=0.98, name="G_bn3")
        self.relu_3 = tf.nn.relu(self.conv_3_bn)
        self.regularization_values.append(reg_3)

        out_dim = 2
        self.conv_4, reg_4 = ops.conv2d(self.relu_3, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_4")
        self.regularization_values.append(reg_4)

        predict = tf.reshape(self.conv_4, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict')

        # Dump prediction out
        predict_real = tf.slice(predict, begin=[0,0,0,0], size=[1,-1,-1,1], name='G_Slice_real_input')
        predict_imag = tf.slice(predict, begin=[0,0,0,1], size=[1,-1,-1,1], name='G_Slice_imag_input')
        tf.summary.image('G_predict_real', predict_real, collections='G')
        tf.summary.image('G_predict_imag', predict_imag, collections='G')

        return predict  # Sum the reg term in the loss

    def __D__(self, input_d, input_type):
        """
        Define the discriminator
        """
        # Dump input image out
        input_real = tf.slice(input_d, begin=[0,0,0,0], size=[1,-1,-1,1], name='D_Slice_real_input')
        input_imag = tf.slice(input_d, begin=[0,0,0,1], size=[1,-1,-1,1], name='D_Slice_imag_input')
        tf.summary.image('D_x_input_real' + input_type, input_real, collections='D')
        tf.summary.image('D_x_input_imag' + input_type, input_imag, collections='D')

        # Model convolutions
        out_dim = 8  # 128x128
        self.conv_1_d, reg_1_d = ops.conv2d(input_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_1")
        self.pool_1_d = tf.nn.max_pool(self.conv_1_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="D_pool_1")
        self.conv_1_bn_d = ops.batch_norm(self.pool_1_d, self.train_phase, decay=0.98, name="D_bn1")
        # self.relu_1_d = tf.nn.relu(self.conv_1_bn_d)
        self.relu_1_d = ops.lrelu(self.conv_1_bn_d)
        self.regularization_values_d.append(reg_1_d)

        out_dim = 16  # 64x64
        self.conv_2_d, reg_2_d = ops.conv2d(self.relu_1_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_2")
        self.pool_2_d = tf.nn.max_pool(self.conv_2_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="D_pool_2")
        self.conv_2_bn_d = ops.batch_norm(self.pool_2_d, self.train_phase, decay=0.98, name="D_bn2")
        # self.relu_2_d = tf.nn.relu(self.conv_2_bn_d)
        self.relu_2_d = ops.lrelu(self.conv_2_bn_d)
        self.regularization_values_d.append(reg_2_d)

        out_dim = 32  # 32x32
        self.conv_3_d, reg_3_d = ops.conv2d(self.relu_2_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_3")
        self.pool_3_d = tf.nn.max_pool(self.conv_3_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="D_pool_3")
        self.conv_3_bn_d = ops.batch_norm(self.pool_3_d, self.train_phase, decay=0.98, name="D_bn3")
        # self.relu_3_d = tf.nn.relu(self.conv_3_bn_d)
        self.relu_3_d = ops.lrelu(self.conv_3_bn_d)
        self.regularization_values_d.append(reg_3_d)

        out_dim = 16  # 16x16
        self.conv_4_d, reg_4_d = ops.conv2d(self.relu_3_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_4")
        self.pool_4_d = tf.nn.max_pool(self.conv_4_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="D_pool_4")
        self.conv_4_bn_d = ops.batch_norm(self.pool_4_d, self.train_phase, decay=0.98, name="D_bn4")
        # self.relu_4_d = tf.nn.relu(self.conv_4_bn_d)
        self.relu_4_d = ops.lrelu(self.conv_4_bn_d)
        self.regularization_values_d.append(reg_4_d)

        out_dim = 1
        self.affine_1_d = ops.linear(tf.contrib.layers.flatten(self.relu_4_d), output_size=out_dim, scope="D_affine_1")
        predict_d = self.affine_1_d
        # Dump prediction out

        return tf.nn.sigmoid(predict_d), predict_d

    def __loss__(self, FLAGS):
        """
        Calculate loss
        :return:
        """
        # regularization ?

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.predict_d_logits,
                                                    tf.ones_like(self.predict_d)))
        tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.predict_d_logits_for_g,
                                                    tf.zeros_like(self.predict_d_for_g)))
        tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')

        self.d_loss = self.d_loss_real + self.d_loss_fake
        tf.summary.scalar('d_loss', self.d_loss, collections='D')

        if len(self.regularization_values_d) > 0:
            reg_loss_d = self.reg_w * tf.reduce_sum(self.regularization_values_d)
            self.d_loss += reg_loss_d
            tf.summary.scalar('d_loss_plus_reg', self.d_loss, collections='D')
            tf.summary.scalar('d_loss_reg_only', reg_loss_d, collections='D')

        # Generative loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.predict_d_logits_for_g,
                                                    tf.ones_like(self.predict_d_for_g)))
        tf.summary.scalar('g_loss', g_loss, collections='G')

        l2_loss = tf.reduce_mean(tf.square(tf.squeeze(self.predict_g) - self.labels), name='L2-Loss')
        tf.summary.scalar('g_loss_L2_only', l2_loss, collections='G')

        self.g_loss = FLAGS.gen_loss_adversarial * g_loss + FLAGS.gen_loss_context * l2_loss
        tf.summary.scalar('g_loss_plus_L2', self.g_loss, collections='G')

        if len(self.regularization_values) > 0:
            reg_loss_g = self.reg_w * tf.reduce_sum(self.regularization_values)
            self.g_loss += reg_loss_g
            tf.summary.scalar('g_loss_plus_L2_plus_reg', self.d_loss, collections='G')
            tf.summary.scalar('g_loss_reg_only', reg_loss_g, collections='D')

        tf.summary.scalar('diff-loss', tf.abs(self.d_loss - self.g_loss), collections='G')

    def __training__(self, learning_rate):
        """
        :param learning_rate:
        :return:

        """
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D_' in var.name]
        self.g_vars = [var for var in t_vars if 'G_' in var.name]

        # Create Adam optimizer with the given learning rate.
        optimizer_d = tf.train.AdamOptimizer(learning_rate)
        optimizer_g = tf.train.AdamOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step_d = tf.Variable(0, name='global_step_d', trainable=False)
        global_step_g = tf.Variable(0, name='global_step_g', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        grad_d = optimizer_d.compute_gradients(loss=self.d_loss, var_list=self.d_vars)
        grad_g = optimizer_g.compute_gradients(loss=self.g_loss, var_list=self.g_vars)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Ensures that we execute the update_ops before performing the train_step
        with tf.control_dependencies(self.update_ops):
            train_op_d = optimizer_d.apply_gradients(grad_d, global_step=global_step_d)
            train_op_g = optimizer_g.apply_gradients(grad_g, global_step=global_step_g)

        return train_op_d, train_op_g

    def __evaluation__(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """

        evalu = tf.reduce_mean(tf.square(tf.squeeze(predict) - tf.squeeze(labels)))
        # tf.summary.scalar("Evaluation", evalu)
        # Return the number of true entries.
        return evalu