import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolutionGAN(BasicModel):
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
        BasicModel.__init__(self, input=input, labels=labels, dims_in=dims_in, dims_out=dims_out)
        # HACK
        self.input_d = labels
        # HACK

        self.FLAGS = FLAGS
        self.train_phase = train_phase
        self.predict_g = None
        self.adv_loss_w = adv_loss_w

        self.predict_d = None
        self.predict_d_logits = None

        self.predict_d_for_g = None
        self.predict_d_logits_for_g = None

        self.train_op_g = None
        self.train_op_d = None

        self.evaluation = None
        self.update_ops = None
        self.x_input_upscale = {}
        self.debug = None
        self.batch_size = self.FLAGS.mini_batch_size
        self.reg_w = self.FLAGS.regularization_weight
        self.regularization_values = []
        self.regularization_sum = None

        self.regularization_values_d = []
        self.regularization_sum_d = None

        tf.get_collection('D')
        tf.get_collection('G')

    def build(self):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g = self.__G__()

        with tf.name_scope('D_'):
            self.predict, self.predict_logits = self.__D__([self.input_d, self.predict_g], input_type="Real")
            self.predict_d, self.predict_d_for_g = tf.split(value=self.predict, num_or_size_splits=2, axis=0)
            self.predict_d_logits, self.predict_d_logits_for_g = tf.split(value=self.predict_logits, num_or_size_splits=2, axis=0)

            # self.predict_d, self.predict_d_logits
            # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            #     self.predict_d_for_g, self.predict_d_logits_for_g = self.__D__(self.predict_g, input_type="Gen")

            if len(self.regularization_values_d) > 0:
                self.regularization_sum_d = sum(self.regularization_values_d)

        with tf.name_scope('loss'):
            # self.loss_g = self.__loss_g__(predict=self.predict_g, self.labels, reg=self.regularization_sum)
            self.__loss__()

        with tf.name_scope('training'):
            self.train_op_d, self.train_op_g = self.__training__(learning_rate=self.FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy L2 norm
            self.evaluation = self.__evaluation__(predict=self.predict_g, labels=self.labels)

    def __G__(self):
        """
        Define the model
        """

        # Dump input image out
        x_real = tf.reshape(self.input['real'], [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]], name='x_real')
        x_imag = tf.reshape(self.input['imag'], [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]], name='x_imag')

        if self.FLAGS.dump_debug:
            tf.summary.image('G_x_input_real', x_real, collections='G', max_outputs=2)
            tf.summary.image('G_x_input_imag', x_imag, collections='G', max_outputs=2)
            tf.summary.image('G_mask', self.labels['mask'], collections='G', max_outputs=1)

        # Apply image resize
        x_real_upscale = tf.image.resize_bilinear(x_real, np.array([self.dims_out[0],self.dims_out[1]]), align_corners=None, name='x_real_upscale')
        x_imag_upscale = tf.image.resize_bilinear(x_imag, np.array([self.dims_out[0],self.dims_out[1]]), align_corners=None, name='x_imag_upscale')

        if self.FLAGS.dump_debug:
            tf.summary.image('G_reconstruct_bilinear', self.get_reconstructed_image(real=x_real_upscale,
                             imag=x_imag_upscale, name='rec_upscale'), collections='G', max_outputs=2)

        self.x_input_upscale['real'] = x_real_upscale
        self.x_input_upscale['imag'] = x_imag_upscale
        # Apply image resize for debugging
        # self.x_input_upscale = tf.image.resize_bilinear(self.input, np.array([self.dims_out[0],
        #                                                               self.dims_out[1]]), align_corners=None,
        #                                                 name='G_x_input_upscale_iamg')

        # Model convolutions
        with tf.name_scope('real'):
            # out_dim = 8
            # self.conv_1, reg_1 = ops.conv2d(x_real_upscale, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_1")
            # self.conv_1_bn = ops.batch_norm(self.conv_1, self.train_phase, decay=0.98, name="G_bn1")
            # self.relu_1 = tf.nn.relu(self.conv_1_bn)
            # self.regularization_values.append(reg_1)

            # # deconv for get bigger image
            # out_dim = 16
            # self.conv_2, reg_2 = ops.conv2d(self.relu_1, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_2")
            # self.conv_2_bn = ops.batch_norm(self.conv_2, self.train_phase, decay=0.98, name="G_bn2")
            # self.relu_2 = tf.nn.relu(self.conv_2_bn)
            # self.regularization_values.append(reg_2)

            # out_dim = 32
            # self.conv_3, reg_3 = ops.conv2d(self.relu_2, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_3")
            # self.conv_3_bn = ops.batch_norm(self.conv_3, self.train_phase, decay=0.98, name="G_bn3")
            # self.relu_3 = tf.nn.relu(self.conv_3_bn)
            # self.regularization_values.append(reg_3)


            # out_dim = 16
            # self.conv_4, reg_4 = ops.conv2d(self.relu_3, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_4")
            # self.conv_4_bn = ops.batch_norm(self.conv_4, self.train_phase, decay=0.98, name="G_bn4")
            # self.relu_4 = tf.nn.relu(self.conv_4_bn)
            # self.regularization_values.append(reg_4)

            # out_dim = 8
            # self.conv_5, reg_5 = ops.conv2d(self.relu_4, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_5")
            # self.conv_5_bn = ops.batch_norm(self.conv_5, self.train_phase, decay=0.98, name="G_bn5")
            # self.relu_5 = tf.nn.relu(self.conv_5_bn)
            # self.regularization_values.append(reg_5)

            # out_dim = 1
            # self.conv_6, reg_6 = ops.conv2d(self.relu_5, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_6")
            # self.regularization_values.append(reg_6)

            out_dim = 16
            self.conv_1, reg_1 = ops.conv2d(x_real_upscale, output_dim=out_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="G_conv_1")
            self.conv_1_bn = ops.batch_norm(self.conv_1, self.train_phase, decay=0.98, name="G_bn1")
            self.relu_1 = tf.nn.relu(self.conv_1_bn)
            self.regularization_values.append(reg_1)
            
            out_dim = 8
            self.conv_2, reg_2 = ops.conv2d(self.relu_1, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_2")
            self.conv_2_bn = ops.batch_norm(self.conv_2, self.train_phase, decay=0.98, name="G_bn2")
            self.relu_2 = tf.nn.relu(self.conv_2_bn)
            self.regularization_values.append(reg_2)
            
            out_dim = 4
            self.conv_3, reg_3 = ops.conv2d(self.relu_2, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_3")
            self.conv_3_bn = ops.batch_norm(self.conv_3, self.train_phase, decay=0.98, name="G_bn3")
            self.relu_3 = tf.nn.relu(self.conv_3_bn)
            self.regularization_values.append(reg_3)
            
            out_dim = 1
            self.conv_4, reg_4 = ops.conv2d(self.relu_3, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_conv_4")
            self.conv_4_bn = ops.batch_norm(self.conv_4, self.train_phase, decay=0.98, name="G_bn4")
            self.relu_4 = tf.nn.relu(self.conv_4_bn)
            self.regularization_values.append(reg_4)
            
            out_dim = 1
            self.conv_5, reg_5 = ops.conv2d(self.relu_4, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_5")
            self.regularization_values.append(reg_5)
            #

        # ## IMAGe
        # # Model convolutions
        with tf.name_scope('imaginary'):
            # out_dim = 8
            # self.conv_11, reg_11 = ops.conv2d(x_imag_upscale, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_11")
            # self.conv_11_bn = ops.batch_norm(self.conv_11, self.train_phase, decay=0.98, name="G_bn11")
            # self.relu_11 = tf.nn.relu(self.conv_11_bn)
            # self.regularization_values.append(reg_11)

            # # deconv for get bigger image
            # out_dim = 16
            # self.conv_22, reg_22 = ops.conv2d(self.relu_11, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_22")
            # self.conv_22_bn = ops.batch_norm(self.conv_22, self.train_phase, decay=0.98, name="G_bn22")
            # self.relu_22 = tf.nn.relu(self.conv_22_bn)
            # self.regularization_values.append(reg_22)

            # out_dim = 32
            # self.conv_33, reg_33 = ops.conv2d(self.relu_22, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_33")
            # self.conv_33_bn = ops.batch_norm(self.conv_33, self.train_phase, decay=0.98, name="G_bn33")
            # self.relu_33 = tf.nn.relu(self.conv_33_bn)
            # self.regularization_values.append(reg_33)


            # out_dim = 16
            # self.conv_44, reg_44 = ops.conv2d(self.relu_33, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_44")
            # self.conv_44_bn = ops.batch_norm(self.conv_44, self.train_phase, decay=0.98, name="G_bn44")
            # self.relu_44 = tf.nn.relu(self.conv_44_bn)
            # self.regularization_values.append(reg_44)

            # out_dim = 8
            # self.conv_55, reg_55 = ops.conv2d(self.relu_44, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_55")
            # self.conv_55_bn = ops.batch_norm(self.conv_55, self.train_phase, decay=0.98, name="G_bn55")
            # self.relu_55 = tf.nn.relu(self.conv_55_bn)
            # self.regularization_values.append(reg_55)

            # out_dim = 1
            # self.conv_66, reg_66 = ops.conv2d(self.relu_55, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_66")
            # self.regularization_values.append(reg_66)

            out_dim = 16
            self.conv_11, reg_11 = ops.conv2d(x_imag_upscale, output_dim=out_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="G_conv_11")
            self.conv_11_bn = ops.batch_norm(self.conv_11, self.train_phase, decay=0.98, name="G_bn11")
            self.relu_11 = tf.nn.relu(self.conv_11_bn)
            self.regularization_values.append(reg_11)
            
            out_dim = 8
            self.conv_22, reg_22 = ops.conv2d(self.relu_11, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_22")
            self.conv_22_bn = ops.batch_norm(self.conv_22, self.train_phase, decay=0.98, name="G_bn22")
            self.relu_22 = tf.nn.relu(self.conv_22_bn)
            self.regularization_values.append(reg_22)
            
            out_dim = 4
            self.conv_33, reg_33 = ops.conv2d(self.relu_22, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_33")
            self.conv_33_bn = ops.batch_norm(self.conv_33, self.train_phase, decay=0.98, name="G_bn33")
            self.relu_33 = tf.nn.relu(self.conv_33_bn)
            self.regularization_values.append(reg_33)
            
            out_dim = 1
            self.conv_44, reg_44 = ops.conv2d(self.relu_33, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_conv_44")
            self.conv_44_bn = ops.batch_norm(self.conv_44, self.train_phase, decay=0.98, name="G_bn44")
            self.relu_44 = tf.nn.relu(self.conv_44_bn)
            self.regularization_values.append(reg_44)
            
            out_dim = 1
            self.conv_55, reg_55 = ops.conv2d(self.relu_44, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_55")
            self.regularization_values.append(reg_55)

        # stack = tf.squeeze(tf.stack([self.conv_6, self.conv_66], axis=3))
        predict = {}
        # predict['real'] = tf.reshape(self.conv_6, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_real')
        # predict['imag'] = tf.reshape(self.conv_66, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_imag')
        predict['real'] = tf.reshape(self.conv_5, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_real')
        predict['imag'] = tf.reshape(self.conv_55, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_imag')

        # Masking
        mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask'], tf.bool)), tf.float32)
        predict['real'] = tf.multiply(predict['real'], mask_not)
        predict['imag'] = tf.multiply(predict['imag'], mask_not)

        input_masked_real = tf.multiply(self.labels['real'], self.labels['mask'], name='input_masked_real')
        input_masked_imag = tf.multiply(self.labels['imag'], self.labels['mask'], name='input_masked_imag')

        with tf.name_scope("final_predict"):
            predict['real'] = tf.add(predict['real'], input_masked_real, name='real')
            predict['imag'] = tf.add(predict['imag'], input_masked_imag, name='imag')

        tf.add_to_collection("predict", predict['real'])
        tf.add_to_collection("predict", predict['imag'])

        # Dump prediction out
        if self.FLAGS.dump_debug:
            tf.summary.image('G_predict_real', predict['real'], collections='G')
            tf.summary.image('G_predict_imag', predict['imag'], collections='G')

        # return tf.stack([predict['real'], predict['imag']], axis=3)

        return predict

    def __D__(self, input_d, input_type):
        """
        Define the discriminator
        """
        # Dump input image out

        input_real = tf.concat(axis=0, values=[input_d[0]['real'], input_d[1]['real']])
        input_imag = tf.concat(axis=0, values=[input_d[0]['imag'], input_d[1]['imag']])

        # Input d holds real&imaginary values. The discriminative decision based on reconstructed image
        input_to_discriminator = self.get_reconstructed_image(real=input_real, imag=input_imag, name='Both')

        org, fake = tf.split(input_to_discriminator, num_or_size_splits=2, axis=0)
        tf.summary.image('D_x_input_reconstructed' + 'Original', org, collections='D', max_outputs=2)
        tf.summary.image('D_x_input_reconstructed' + 'Fake', fake, collections='G', max_outputs=2)

        # Model convolutions
        out_dim = 8  # 128x128
        self.conv_1_d, reg_1_d = ops.conv2d(input_to_discriminator, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_1")
        self.pool_1_d = tf.nn.max_pool(self.conv_1_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name="D_pool_1")
        self.conv_1_bn_d = ops.batch_norm(self.pool_1_d, self.train_phase, decay=0.98, name="D_bn1")
        # self.relu_1_d = tf.nn.relu(self.conv_1_bn_d)
        self.relu_1_d = ops.lrelu(self.conv_1_bn_d)
        self.regularization_values_d.append(reg_1_d)

        out_dim = 16  # 64x64
        self.conv_2_d, reg_2_d = ops.conv2d(self.relu_1_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_2")
        self.pool_2_d = tf.nn.max_pool(self.conv_2_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name="D_pool_2")
        self.conv_2_bn_d = ops.batch_norm(self.pool_2_d, self.train_phase, decay=0.98, name="D_bn2")
        # self.relu_2_d = tf.nn.relu(self.conv_2_bn_d)
        self.relu_2_d = ops.lrelu(self.conv_2_bn_d)
        self.regularization_values_d.append(reg_2_d)

        # out_dim = 32  # 32x32
        out_dim = 16  # 32x32
        self.conv_3_d, reg_3_d = ops.conv2d(self.relu_2_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_3")
        self.pool_3_d = tf.nn.max_pool(self.conv_3_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name="D_pool_3")
        self.conv_3_bn_d = ops.batch_norm(self.pool_3_d, self.train_phase, decay=0.98, name="D_bn3")
        # self.relu_3_d = tf.nn.relu(self.conv_3_bn_d)
        self.relu_3_d = ops.lrelu(self.conv_3_bn_d)
        self.regularization_values_d.append(reg_3_d)

        # out_dim = 16  # 16x16
        # self.conv_4_d, reg_4_d = ops.conv2d(self.relu_3_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
        #                                     name="D_conv_4")
        # self.pool_4_d = tf.nn.max_pool(self.conv_4_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
        #                                name="D_pool_4")
        # self.conv_4_bn_d = ops.batch_norm(self.pool_4_d, self.train_phase, decay=0.98, name="D_bn4")
        # # self.relu_4_d = tf.nn.relu(self.conv_4_bn_d)
        # self.relu_4_d = ops.lrelu(self.conv_4_bn_d)
        # self.regularization_values_d.append(reg_4_d)

        out_dim = 1
        self.affine_1_d = ops.linear(tf.contrib.layers.flatten(self.relu_3_d), output_size=out_dim, scope="D_affine_1")
        # self.affine_1_d = ops.linear(tf.contrib.layers.flatten(self.relu_4_d), output_size=out_dim, scope="D_affine_1")
        predict_d = self.affine_1_d
        # Dump prediction out

        return tf.nn.sigmoid(predict_d), predict_d

    def __loss__(self):
        """
        Calculate loss
        :return:
        """
        # regularization ?

        self.d_loss_real = tf.reduce_mean(ops.binary_cross_entropy(preds=self.predict_d, targets=tf.ones_like(self.predict_d)))
            # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits,
            #                                         labels=tf.ones_like(self.predict_d)))

        tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')

        self.d_loss_fake = tf.reduce_mean(ops.binary_cross_entropy(preds=self.predict_d_for_g, targets=tf.zeros_like(self.predict_d_for_g)))
            # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits_for_g,
            #                                         labels=tf.zeros_like(self.predict_d_for_g)))

        tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')

        self.d_loss = self.d_loss_real + self.d_loss_fake
        tf.summary.scalar('d_loss', self.d_loss, collections='D')

        if len(self.regularization_values_d) > 0:
            reg_loss_d = self.reg_w * tf.reduce_sum(self.regularization_values_d)
            self.d_loss += reg_loss_d
            if self.FLAGS.dump_debug:
                tf.summary.scalar('d_loss_plus_reg', self.d_loss, collections='D')
                tf.summary.scalar('d_loss_reg_only', reg_loss_d, collections='D')

        # Generative loss
        g_loss = tf.reduce_mean(ops.binary_cross_entropy(preds=self.predict_d_for_g, targets=tf.ones_like(self.predict_d_for_g)))
            # tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits_for_g,
            #                                         labels=tf.ones_like(self.predict_d_for_g)))

        tf.summary.scalar('g_loss', g_loss, collections='G')

        # Context loss L2
        mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask'], tf.bool)), tf.float32)
        real_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g['real'] - self.labels['real'], mask_not))
        imag_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g['imag'] - self.labels['imag'], mask_not))
        self.context_loss = tf.reduce_mean(tf.square(real_diff) + tf.square(imag_diff), name='Context_loss_mean')

        # L1
        # mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask'], tf.bool)), tf.float32)
        # real_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g['real'] - self.labels['real'], mask_not))
        # imag_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g['imag'] - self.labels['imag'], mask_not))
        # self.context_loss = tf.reduce_mean(tf.abs(real_diff) + tf.abs(imag_diff), name='Context_loss_mean')

        # L2, on FFT
        # rec_diff = self.get_reconstructed_image(self.predict_g['real'], self.predict_g['imag'], name='1')\
        #            - self.get_reconstructed_image(self.labels['real'], self.labels['imag'], name='2')
        # self.context_loss = tf.reduce_mean(tf.square(rec_diff), name='Context_loss_mean')

        tf.summary.scalar('g_loss_context_only', self.context_loss, collections='G')

        self.g_loss = self.adv_loss_w * g_loss + self.FLAGS.gen_loss_context * self.context_loss
        # self.g_loss = self.FLAGS.gen_loss_adversarial * g_loss + self.FLAGS.gen_loss_context * context_loss
        tf.summary.scalar('g_loss_plus_context', self.g_loss, collections='G')

        if len(self.regularization_values) > 0:
            reg_loss_g = self.reg_w * tf.reduce_sum(self.regularization_values)
            self.g_loss += reg_loss_g
            if self.FLAGS.dump_debug:
                tf.summary.scalar('g_loss_plus_context_plus_reg', self.g_loss, collections='G')
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
        # evalu = tf.reduce_mean(tf.square(tf.squeeze(predict['real']) - tf.squeeze(labels['real']))) \
        #         + tf.reduce_mean(tf.square(tf.squeeze(predict['imag']) - tf.squeeze(labels['imag'])))
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
        rec_image_complex = tf.expand_dims(tf.ifft2d(complex_k_space_label), axis=3)
        rec_image = tf.reshape(tf.abs(rec_image_complex), shape=[-1, 256, 256, 1])

        # Shifting
        top, bottom = tf.split(rec_image, num_or_size_splits=2, axis=1)
        top_left, top_right = tf.split(top, num_or_size_splits=2, axis=2)
        bottom_left, bottom_right = tf.split(bottom, num_or_size_splits=2, axis=2)

        top_shift = tf.concat(axis=2, values=[bottom_right, bottom_left])
        bottom_shift = tf.concat(axis=2, values=[top_right, top_left])
        shifted_image = tf.concat(axis=1, values=[top_shift, bottom_shift])
        return shifted_image
