import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolutionWGAN(BasicModel):
    """
    Represents k-space super resolution model
    """

    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, FLAGS=None, train_phase=None, adv_loss_w=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=labels, labels=labels, dims_in=dims_in, dims_out=dims_out)
        self.input = labels
        print("Input is y label")
        # HACK
        self.input_d = labels
        # HACK

        self.FLAGS = FLAGS
        self.train_phase = train_phase
        self.predict_g = None
        self.adv_loss_w = adv_loss_w

        self.train_op_g = None
        self.train_op_d = None

        self.evaluation = None
        self.update_ops = None
        self.x_input_upscale = {}
        self.debug = None
        self.batch_size = self.FLAGS.mini_batch_size
        # self.reg_w = self.FLAGS.regularization_weight
        self.regularization_values = []
        self.regularization_sum = None

        self.regularization_values_d = []
        self.regularization_sum_d = None

        self.clip_weights = None

        tf.get_collection('G')

    def build(self):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g = self.__G__()

        with tf.name_scope('loss'):
            # self.loss_g = self.__loss_g__(predict=self.predict_g, self.labels, reg=self.regularization_sum)
            self.__loss__()

        with tf.name_scope('training'):
            self.train_op_g = self.__training__(learning_rate=self.FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy L2 norm
            self.evaluation = self.__evaluation__(predict=self.predict_g, labels=self.labels)

    def __G__(self):
        """
        Define the model
        """
        x_real = self.input['real'] * self.input['mask']
        x_imag = self.input['imag'] * self.input['mask']

        # if self.FLAGS.gen_loss_adversarial > 0:
        #     mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask'], tf.bool)), tf.float32)
        #     print "Noise level: (-0.01,0.01)"
        #     minval = -0.01
        #     maxval = 0.01
        #     noise_real = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval,
        #                                                 dtype=tf.float32, seed=None, name='z_real')
        #     noise_imag = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval,
        #                                                 dtype=tf.float32, seed=None, name='z_imag')
        #     x_real += noise_real
        #     x_imag += noise_imag

        # input image is zero padded image
        input_image = self.get_reconstructed_image(real=x_real, imag=x_imag, name='Both')

        # Create the inputs
        ref_image = self.get_reconstructed_image(real=self.input['real'], imag=self.input['imag'], name='Both')
        self.ref_image = ref_image

        ref_image = tf.abs(tf.complex(real=tf.expand_dims(ref_image[:,0,:,:], axis=1), imag=tf.expand_dims(ref_image[:,1,:,:], axis=1)))
        tf.summary.image('G_reference', tf.transpose(ref_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        if self.FLAGS.dump_debug:
            zero_image = tf.abs(tf.complex(real=tf.expand_dims(input_image[:, 0, :, :], axis=1),
                                          imag=tf.expand_dims(input_image[:, 1, :, :], axis=1)))
            tf.summary.image('G_zero', tf.transpose(zero_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        self.x_input_upscale['real'] = x_real
        self.x_input_upscale['imag'] = x_imag

        # Model convolutions
        # Model convolutions
        out_dim = 8  # 256x256 => 128x128
        conv1, pool1 = ops.conv_conv_pool(input_image, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                          training=self.train_phase, name='G_block_1', norm=False)

        out_dim = 16  # 128x128 => 64x64
        conv2, pool2 = ops.conv_conv_pool(pool1, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                          training=self.train_phase, name='G_block_2', norm=False)

        out_dim = 32  # 64x128 => 32x32
        conv3, pool3 = ops.conv_conv_pool(pool2, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                          training=self.train_phase, name='G_block_3', norm=False)

        out_dim = 64  # 32x32 => 16x16
        conv4, pool4 = ops.conv_conv_pool(pool3, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                          training=self.train_phase, name='G_block_4', norm=False)

        out_dim = 128  # 16x16
        conv5 = ops.conv_conv_pool(pool4, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                   training=self.train_phase, name='G_block_5', pool=False, norm=False)

        # concat (16x2)x(16x2) with 32x32
        up6 = ops.upsample_concat(inputA=conv5, input_B=conv4, name='G_block_6')
        out_dim = 64
        conv6 = ops.conv_conv_pool(up6, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                   training=self.train_phase, name='G_block_6', pool=False, norm=False)

        # concat (32x2)x(32x2) with 64x64
        up7 = ops.upsample_concat(inputA=conv6, input_B=conv3, name='G_block_7')
        out_dim = 32
        conv7 = ops.conv_conv_pool(up7, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                   training=self.train_phase, name='G_block_7', pool=False, norm=False)

        # concat (64x2)x(64x2) with 128x128
        up8 = ops.upsample_concat(inputA=conv7, input_B=conv2, name='G_block_8')
        out_dim = 16
        conv8 = ops.conv_conv_pool(up8, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                   training=self.train_phase, name='G_block_8', pool=False, norm=False)

        # concat (128x2)x(128x2) with 256x256
        up9 = ops.upsample_concat(inputA=conv8, input_B=conv1, name='G_block_9')
        out_dim = 8
        conv9 = ops.conv_conv_pool(up9, n_filters=[out_dim, out_dim], activation=tf.nn.relu,
                                   training=self.train_phase, name='G_block_9', pool=False, norm=False)

        # reduce - 256x256x8 -> 256x256x2
        out_dim = 2
        conv_last = ops.conv2d(conv9, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_conv_last")

        predict = {}
        predict['real'] = tf.reshape(conv_last[:,0,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_real')
        predict['imag'] = tf.reshape(conv_last[:,1,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_imag')

        with tf.name_scope("final_predict"):
            predict['real'] = tf.add(predict['real'], tf.expand_dims(input_image[:,0,:,:], axis=1), name='real')
            predict['imag'] = tf.add(predict['imag'], tf.expand_dims(input_image[:,1,:,:], axis=1), name='imag')

        tf.add_to_collection("predict", predict['real'])
        tf.add_to_collection("predict", predict['imag'])

        # Dump prediction out
        out_image = tf.abs(tf.complex(real=predict['real'], imag=predict['imag']))
        tf.summary.image('G_predict', tf.transpose(out_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        if self.FLAGS.dump_debug:
            tf.summary.image('G_predict_real', tf.transpose(predict['real'], (0, 2, 3, 1)), collections='G', max_outputs=4)
            tf.summary.image('G_predict_imag', tf.transpose(predict['imag'], (0, 2, 3, 1)), collections='G', max_outputs=4)
            tf.summary.image('G_input_real', tf.transpose(tf.expand_dims(self.ref_image[:, 0, :, ], axis=1), (0, 2, 3, 1)),
                             collections='G', max_outputs=4)
            tf.summary.image('G_input_imag', tf.transpose(tf.expand_dims(self.ref_image[:, 1, :, ], axis=1), (0, 2, 3, 1)),
                             collections='G', max_outputs=4)
        return predict

    def __loss__(self):
        """
        Calculate loss
        :return:
        """
        # Context loss L2
        label_real = tf.expand_dims(self.ref_image[:,0,:,], axis=1)
        label_imag = tf.expand_dims(self.ref_image[:,1,:,], axis=1)
        real_diff = tf.contrib.layers.flatten(self.predict_g['real'] - label_real)
        imag_diff = tf.contrib.layers.flatten(self.predict_g['imag'] - label_imag)

        print("You are using L2 loss")
        self.context_loss = tf.reduce_mean(tf.square(real_diff) + tf.square(imag_diff), name='Context_loss_mean')

        self.g_loss = self.FLAGS.gen_loss_context * self.context_loss
        tf.summary.scalar('g_loss', self.g_loss, collections='G')

        self.reg_loss_g = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='G')
        self.g_loss_no_reg = self.g_loss
        self.g_loss += self.reg_loss_g
        if self.FLAGS.dump_debug:
            tf.summary.scalar('g_loss_plus_reg', self.g_loss, collections='G')
            tf.summary.scalar('g_loss_reg_only', self.reg_loss_g, collections='D')

    def __training__(self, learning_rate):
        """
        :param learning_rate:
        :return:

        """
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'G_' in var.name]

        # Create RMSProb optimizer with the given learning rate.
        optimizer_g = tf.train.AdamOptimizer(self.FLAGS.learning_rate)

        # Create a variable to track the global step.
        global_step_g = tf.Variable(0, name='global_step_g', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        grad_g = optimizer_g.compute_gradients(loss=self.g_loss, var_list=self.g_vars)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Ensures that we execute the update_ops before performing the train_step
        with tf.control_dependencies(self.update_ops):
            train_op_g = optimizer_g.apply_gradients(grad_g, global_step=global_step_g)

        return train_op_g

    def __evaluation__(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """
        evalu = self.context_loss
        return evalu

    def get_reconstructed_image(self, real, imag, name=None):
        """
        :param real:
        :param imag:
        :param name:
        :return:
        """
        complex_k_space_label = tf.complex(real=tf.squeeze(real), imag=tf.squeeze(imag), name=name+"_complex_k_space")
        rec_image_complex = tf.expand_dims(tf.ifft2d(complex_k_space_label), axis=1)
        
        rec_image_real = tf.reshape(tf.real(rec_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        rec_image_imag = tf.reshape(tf.imag(rec_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])

        # Shifting
        top, bottom = tf.split(rec_image_real, num_or_size_splits=2, axis=2)
        top_left, top_right = tf.split(top, num_or_size_splits=2, axis=3)
        bottom_left, bottom_right = tf.split(bottom, num_or_size_splits=2, axis=3)

        top_shift = tf.concat(axis=3, values=[bottom_right, bottom_left])
        bottom_shift = tf.concat(axis=3, values=[top_right, top_left])
        shifted_image = tf.concat(axis=2, values=[top_shift, bottom_shift])

        # Shifting
        top_imag, bottom_imag = tf.split(rec_image_imag, num_or_size_splits=2, axis=2)
        top_left_imag, top_right_imag = tf.split(top_imag, num_or_size_splits=2, axis=3)
        bottom_left_imag, bottom_right_imag = tf.split(bottom_imag, num_or_size_splits=2, axis=3)

        top_shift_imag = tf.concat(axis=3, values=[bottom_right_imag, bottom_left_imag])
        bottom_shift_imag = tf.concat(axis=3, values=[top_right_imag, top_left_imag])
        shifted_image_imag = tf.concat(axis=2, values=[top_shift_imag, bottom_shift_imag])

        shifted_image_two_channels = tf.stack([shifted_image[:,0,:,:], shifted_image_imag[:,0,:,:]], axis=1)
        return shifted_image_two_channels

    def get_weights_regularization(self, dump=False, collection=None):
        """
        Calculate sum of regularization (L2)
        :param dump: dump to tensorboard
        :return:
        """
        if collection is None:
            w_collection = tf.get_collection('regularization_w')
            b_collection = tf.get_collection('regularization_b')
        else:
            w_collection = [var for var in tf.get_collection('regularization_w') if collection in var.name]
            b_collection = [var for var in tf.get_collection('regularization_b') if collection in var.name]

        reg_w = tf.add_n(w_collection, name='regularization_w') if len(w_collection) > 0 else 0
        reg_b = tf.add_n(b_collection, name='regularization_b') if len(b_collection) > 0 else 0

        if dump:
            tf.summary.scalar('Regularization - W', reg_w, collections=collection)
            tf.summary.scalar('Regularization - b', reg_b, collections=collection)

        return self.FLAGS.reg_w * reg_w + self.FLAGS.reg_b * reg_b
