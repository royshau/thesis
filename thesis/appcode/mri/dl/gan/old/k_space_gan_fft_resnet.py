import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops
from appcode.mri.dl.gan.k_space_gan_fft import KSpaceSuperResolutionGAN


class KSpaceSuperResolutionResGAN(KSpaceSuperResolutionGAN):
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
        KSpaceSuperResolutionGAN.__init__(self, input=input, labels=labels, dims_in=dims_in,
                                          dims_out=dims_out, FLAGS=FLAGS, train_phase=train_phase, adv_loss_w=adv_loss_w)

    def __G__(self):
        """
        Define the model
        """
        # Apply image resize for debugging
        self.x_input_upscale = tf.image.resize_bilinear(self.input, np.array([self.dims_out[0],
                                                                              self.dims_out[1]]), align_corners=None,
                                                        name='G_x_input_upscale')

        # Dump input image out
        input_real = tf.slice(self.input, begin=[0, 0, 0, 0], size=[1, -1, -1, 1], name='G_Slice_real_input')
        input_imag = tf.slice(self.input, begin=[0, 0, 0, 1], size=[1, -1, -1, 1], name='G_Slice_imag_input')

        tf.summary.image('G_x_input_real', input_real, collections='G')
        tf.summary.image('G_x_input_imag', input_imag, collections='G')

        # out_dim = 16
        # self.res_block_1, res_block_reg = ops.res_block(self.relu_1, output_dim=out_dim, train_phase=self.train_phase,
        #                                                 k_h=3, k_w=3, d_h=1, d_w=1, name="G_res_block_1")
        # self.regularization_values.append(res_block_reg)

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
        predict_real = tf.slice(predict, begin=[0, 0, 0, 0], size=[1, -1, -1, 1], name='G_Slice_real_input')
        predict_imag = tf.slice(predict, begin=[0, 0, 0, 1], size=[1, -1, -1, 1], name='G_Slice_imag_input')
        tf.summary.image('G_predict_real', predict_real, collections='G')
        tf.summary.image('G_predict_imag', predict_imag, collections='G')

        return predict  # Sum the reg term in the loss

    def __D__(self, input_d, input_type):
        """
        Define the discriminator
        """
        # Dump input image out
        input_real = tf.slice(input_d, begin=[0, 0, 0, 0], size=[1, -1, -1, 1], name='D_Slice_real_input')
        input_imag = tf.slice(input_d, begin=[0, 0, 0, 1], size=[1, -1, -1, 1], name='D_Slice_imag_input')
        tf.summary.image('D_x_input_real' + input_type, input_real, collections='D')
        tf.summary.image('D_x_input_imag' + input_type, input_imag, collections='D')

        # Input d holds real&imaginary values. The discriminative decision based on reconstructed image

        input_to_discriminator = self.get_reconstructed_image(real=input_d[:,:,:,0], imag=input_d[:,:,:,1], name=input_type)
        tf.summary.image('D_x_input_reconstructed' + input_type, input_to_discriminator, collections='D')

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

        out_dim = 32  # 32x32
        self.conv_3_d, reg_3_d = ops.conv2d(self.relu_2_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_3")
        self.pool_3_d = tf.nn.max_pool(self.conv_3_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name="D_pool_3")
        self.conv_3_bn_d = ops.batch_norm(self.pool_3_d, self.train_phase, decay=0.98, name="D_bn3")
        # self.relu_3_d = tf.nn.relu(self.conv_3_bn_d)
        self.relu_3_d = ops.lrelu(self.conv_3_bn_d)
        self.regularization_values_d.append(reg_3_d)

        out_dim = 16  # 16x16
        self.conv_4_d, reg_4_d = ops.conv2d(self.relu_3_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_4")
        self.pool_4_d = tf.nn.max_pool(self.conv_4_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name="D_pool_4")
        self.conv_4_bn_d = ops.batch_norm(self.pool_4_d, self.train_phase, decay=0.98, name="D_bn4")
        # self.relu_4_d = tf.nn.relu(self.conv_4_bn_d)
        self.relu_4_d = ops.lrelu(self.conv_4_bn_d)
        self.regularization_values_d.append(reg_4_d)

        out_dim = 1
        self.affine_1_d = ops.linear(tf.contrib.layers.flatten(self.relu_4_d), output_size=out_dim, scope="D_affine_1")
        predict_d = self.affine_1_d
        # Dump prediction out

        return tf.nn.sigmoid(predict_d), predict_d

    def __loss__(self):
        """
        Calculate loss
        :return:
        """
        # regularization ?

        # self.d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits,
        #                                             labels=tf.ones_like(self.predict_d)))
        self.d_loss_real = tf.reduce_mean(
            ops.binary_cross_entropy(preds=self.predict_d, targets=tf.ones_like(self.predict_d)))

        tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')

        # self.d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits_for_g,
        #                                             labels=tf.zeros_like(self.predict_d_for_g)))

        self.d_loss_fake = tf.reduce_mean(
            ops.binary_cross_entropy(preds=self.predict_d_for_g, targets=tf.zeros_like(self.predict_d_for_g)))

        tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')

        self.d_loss = self.d_loss_real + self.d_loss_fake
        tf.summary.scalar('d_loss', self.d_loss, collections='D')

        if len(self.regularization_values_d) > 0:
            reg_loss_d = self.reg_w * tf.reduce_sum(self.regularization_values_d)
            self.d_loss += reg_loss_d
            tf.summary.scalar('d_loss_plus_reg', self.d_loss, collections='D')
            tf.summary.scalar('d_loss_reg_only', reg_loss_d, collections='D')

        # Generative loss
        # g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict_d_logits_for_g,
        #                                             labels=tf.ones_like(self.predict_d_for_g)))
        g_loss = tf.reduce_mean(
            ops.binary_cross_entropy(preds=self.predict_d_for_g, targets=tf.ones_like(self.predict_d_for_g)))

        tf.summary.scalar('g_loss', g_loss, collections='G')

        context_loss = tf.reduce_mean(tf.square(tf.squeeze(self.predict_g) - self.labels), name='L2-Loss')
        tf.summary.scalar('g_loss_context_only', context_loss, collections='G')

        # print("from inside %f" % self.FLAGS.gen_loss_adversarial)
        # self.g_loss = self.FLAGS.gen_loss_adversarial * g_loss + self.FLAGS.gen_loss_context * context_loss

        self.g_loss = self.adb_loss_w * g_loss + self.FLAGS.gen_loss_context * context_loss

        tf.summary.scalar('g_loss_plus_context', self.g_loss, collections='G')

        if len(self.regularization_values) > 0:
            reg_loss_g = self.reg_w * tf.reduce_sum(self.regularization_values)
            self.g_loss += reg_loss_g
            tf.summary.scalar('g_loss_plus_context_plus_reg', self.g_loss, collections='G')
            tf.summary.scalar('g_loss_reg_only', reg_loss_g, collections='D')

        tf.summary.scalar('diff-loss', tf.abs(self.d_loss - self.g_loss), collections='G')