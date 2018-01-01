import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops

# D see the abs(image)
# G Content loss is on the abs image


class KSpaceSuperResolutionGLWGAN(BasicModel):
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
        self.predict_g2 = None
        self.adv_loss_w = adv_loss_w

        self.predict_d = None
        self.predict_d_logits = None

        self.predict_d_for_g = None
        self.predict_d_logits_for_g = None
        self.reconstructed_image_reference = None

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

        tf.get_collection('D')
        tf.get_collection('G')

    def build(self):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g = self.__G__()
            self.predict_g2 = self.__G2__()

        with tf.name_scope('D_'):

            # Create reference examples
            # Input d holds real&imaginary values. The discriminative decision based on reconstructed image
            self.reconstructed_image_reference = self.get_reconstructed_image(real=self.input_d['real'],
                                                                              imag=self.input_d['imag'], name='Both_gt')

            predict_g2_stacked = tf.stack([self.predict_g2['real'][:,0,:,:], self.predict_g2['imag'][:,0,:,:]], axis=1)

            self.predict, self.predict_logits = self.__D__([self.reconstructed_image_reference, predict_g2_stacked])

            self.predict_d, self.predict_d_for_g = tf.split(value=self.predict, num_or_size_splits=2, axis=0)
            self.predict_d_logits, self.predict_d_logits_for_g = tf.split(value=self.predict_logits,
                                                                          num_or_size_splits=2, axis=0)
            self.clip_weights = self.__clip_weights__()

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
        Only for debug, this model already trained an we used it's outputs
        """
        # Create the inputs
        x_real = self.input['real_g']
        x_imag = self.input['imag_g']

        predict = {}
        with tf.name_scope("final_predict"):
            predict['real'] = x_real
            predict['imag'] = x_imag

        tf.add_to_collection("predict", predict['real'])
        tf.add_to_collection("predict", predict['imag'])

        # Dump prediction out
        if self.FLAGS.dump_debug:
            tf.summary.image('G_predict_real_g1', tf.transpose(predict['real'], (0, 2, 3, 1)), collections='G')
            tf.summary.image('G_predict_imag_g1', tf.transpose(predict['imag'], (0, 2, 3, 1)), collections='G')

        return predict

    def __G2__(self):
        """
        This network gets the generator's 1 output (estimated k-space) and
        fine tune the reconstructed image results
        """
        # Input d holds real&imaginary values. The discriminative decision based on reconstructed image
        reconstructed_image = self.get_reconstructed_image(real=self.predict_g['real'], imag=self.predict_g['imag'], name='Both')

        reconstructed_image_to_show = tf.expand_dims(input=tf.complex(real=reconstructed_image[:, 0, :, :],
                                                             imag=reconstructed_image[:, 1, :, :]), dim=1)
        reconstructed_image_to_show = tf.abs(reconstructed_image_to_show)

        tf.summary.image('G2_reconstructed_input' + 'Fake_from_G', tf.transpose(reconstructed_image_to_show, (0, 2, 3, 1)),
                         collections='G2', max_outputs=4)

        # Model convolutions
        # with tf.name_scope('real'):
        out_dim = 32
        conv_1 = ops.conv2d(reconstructed_image, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_1")
        conv_1_bn = ops.batch_norm(conv_1, self.train_phase, decay=0.98, name="G2_bn1")
        relu_1 = tf.nn.relu(conv_1_bn, name='G2_relu1')

        # Pool to 128x128
        conv_2 = ops.conv2d(relu_1, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_2")
        conv_2_bn = ops.batch_norm(conv_2, self.train_phase, decay=0.98, name="G2_bn2")
        pool_2 = tf.layers.max_pooling2d(conv_2_bn, pool_size=[2, 2], strides=2, padding='same',
                                              data_format='channels_first', name="G2_pool_2")
        relu_2 = tf.nn.relu(pool_2, name='G2_relu2')

        # Stay with 128x128
        conv_3 = ops.conv2d(relu_2, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_3")
        conv_3_bn = ops.batch_norm(conv_3, self.train_phase, decay=0.98, name="G2_bn3")
        relu_3 = tf.nn.relu(conv_3_bn, name='G2_relu3')

        # Pool to 64x64
        conv_4 = ops.conv2d(relu_3, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_4")
        conv_4_bn = ops.batch_norm(conv_4, self.train_phase, decay=0.98, name="G2_bn4")
        pool_4 = tf.layers.max_pooling2d(conv_4_bn, pool_size=[2, 2], strides=2, padding='same',
                                              data_format='channels_first', name="G2_pool_4")
        relu_4 = tf.nn.relu(pool_4, name='G2_relu4')

        # Stay with 64x64
        conv_5 = ops.conv2d(relu_4, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_5")
        conv_5_bn = ops.batch_norm(conv_5, self.train_phase, decay=0.98, name="G2_bn5")
        relu_5 = tf.nn.relu(conv_5_bn, name='G2_relu5')

        # Add noise GAN
        print "Noise level: (-0.05,0.05)"
        minval = -0.05
        maxval = 0.05
        noise = tf.random_uniform(shape=tf.shape(relu_5), minval=minval, maxval=maxval, dtype=tf.float32, seed=None, name='noise')
        relu_5 += noise

        # From here, enlarge the network - decoder

        # Enlarge to 128x128
        out_shape = [self.batch_size, out_dim, 128, 128]
        conv_6 = ops.conv2d_transpose(relu_5, output_shape=out_shape, k_h=3, k_w=3, d_h=2, d_w=2, name="G2_conv_6")
        conv_6_bn = ops.batch_norm(conv_6, self.train_phase, decay=0.98, name="G2_bn6")
        relu_6 = tf.nn.relu(conv_6_bn, name='G2_relu6')

        # Stay with 128x128
        conv_7 = ops.conv2d(relu_6, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G2_conv_7")
        conv_7_bn = ops.batch_norm(conv_7, self.train_phase, decay=0.98, name="G2_bn7")
        relu_7 = tf.nn.relu(conv_7_bn, name='G2_relu7')

        # Enlarge to 256x256
        out_shape = [self.batch_size, out_dim, 256, 256]
        conv_8 = ops.conv2d_transpose(relu_7, output_shape=out_shape, k_h=3, k_w=3, d_h=2, d_w=2, name="G2_conv_8")
        conv_8_bn = ops.batch_norm(conv_8, self.train_phase, decay=0.98, name="G2_bn8")
        relu_8 = tf.nn.relu(conv_8_bn, name='G2_relu8')

        out_dim = 2
        conv_last = ops.conv2d(relu_8, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G2_conv_last")

        predict = {}
        predict['real'] = tf.reshape(conv_last[:,0,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G2_predict_real')
        predict['imag'] = tf.reshape(conv_last[:,1,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G2_predict_imag')

        tf.add_to_collection("predict", predict['real'])
        tf.add_to_collection("predict", predict['imag'])

        # Dump prediction out
        if self.FLAGS.dump_debug:
            g2_image_to_show = tf.complex(real=predict['real'], imag=predict['imag'])
            g2_image_to_show = tf.abs(g2_image_to_show)
            tf.summary.image('G2_reconstructed_output' + 'Fake_from_G',
                             tf.transpose(g2_image_to_show, (0, 2, 3, 1)),
                             collections='G2', max_outputs=4)
        return predict

    def __D__(self, input_d):
        """
        Define the discriminator
        """
        # Input d holds real&imaginary values. The discriminative decision based on reconstructed image
        input_to_discriminator = input_d
        org = input_to_discriminator[0]
        fake = input_to_discriminator[1]

        rec_org = tf.abs(tf.expand_dims(input=tf.complex(real=org[:, 0, :, :], imag=org[:, 1, :, :]), dim=1))
        rec_fake = tf.abs(tf.expand_dims(input=tf.complex(real=fake[:, 0, :, :], imag=fake[:, 1, :, :]), dim=1))
        tf.summary.image('D_x_input_reconstructed' + 'Original', tf.transpose(rec_org, (0,2,3,1)), collections='D', max_outputs=4)
        tf.summary.image('D_x_input_reconstructed' + 'Fake', tf.transpose(rec_fake, (0,2,3,1)), collections='G2', max_outputs=4)
        input_to_discriminator = tf.concat(input_to_discriminator, axis=0)

        # Let the Discriminator see the final result:
        input_to_discriminator = tf.abs(tf.expand_dims(input=tf.complex(real=input_to_discriminator[:,0,:,:],
                                                                        imag=input_to_discriminator[:,1,:,:]), dim=1))
        # Model convolutions
        out_dim = 8  # 128x128
        self.conv_1_d = ops.conv2d(input_to_discriminator, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_1")
        self.pool_1_d = tf.layers.max_pooling2d(self.conv_1_d, pool_size=[2, 2], strides=2, padding='same',
                                              data_format='channels_first',name="D_pool_1")
        self.conv_1_bn_d = ops.batch_norm(self.pool_1_d, self.train_phase, decay=0.98, name="D_bn1")
        # self.relu_1_d = tf.nn.relu(self.conv_1_bn_d)
        self.relu_1_d = ops.lrelu(self.conv_1_bn_d, name="D_relu1")

        out_dim = 16  # 64x64
        self.conv_2_d = ops.conv2d(self.relu_1_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_2")
        self.pool_2_d = tf.layers.max_pooling2d(self.conv_2_d, pool_size=[2, 2], strides=2, padding='same',
                                              data_format='channels_first',name="D_pool_2")
        self.conv_2_bn_d = ops.batch_norm(self.pool_2_d, self.train_phase, decay=0.98, name="D_bn2")
        # self.relu_2_d = tf.nn.relu(self.conv_2_bn_d)
        self.relu_2_d = ops.lrelu(self.conv_2_bn_d, name="D_relu2")

        # out_dim = 32  # 32x32
        out_dim = 8  # 32x32
        self.conv_3_d = ops.conv2d(self.relu_2_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                                            name="D_conv_3")
        self.pool_3_d = tf.layers.max_pooling2d(self.conv_3_d, pool_size=[2, 2], strides=2, padding='same',
                                              data_format='channels_first',name="D_pool_3")
        self.conv_3_bn_d = ops.batch_norm(self.pool_3_d, self.train_phase, decay=0.98, name="D_bn3")
        # self.relu_3_d = tf.nn.relu(self.conv_3_bn_d)
        self.relu_3_d = ops.lrelu(self.conv_3_bn_d, name="D_relu3")

        # out_dim = 16  # 16x16
        # self.conv_4_d = ops.conv2d(self.relu_3_d, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
        #                                     name="D_conv_4")
        #self.pool_4_d = tf.layers.max_pooling2d(self.conv_4_d, pool_size=[2, 2], strides=2, padding='same',
        #                                      data_format='channels_first',name="D_pool_4")
        # self.conv_4_bn_d = ops.batch_norm(self.pool_4_d, self.train_phase, decay=0.98, name="D_bn4")
        # # self.relu_4_d = tf.nn.relu(self.conv_4_bn_d)
        # self.relu_4_d = ops.lrelu(self.conv_4_bn_d)

        out_dim = 1
        self.affine_1_d = ops.linear(tf.contrib.layers.flatten(self.relu_3_d), output_size=out_dim, scope="D_affine_1")
        predict_d = self.affine_1_d
        # Dump prediction out

        return tf.nn.sigmoid(predict_d), predict_d

    def __loss__(self):
        """
        Calculate loss
        :return:
        """
        with tf.variable_scope("discriminator") as scope:
            self.d_loss_real = tf.reduce_mean(self.predict_d_logits)
            tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')
            # scope.reuse_variables()
            self.d_loss_fake = tf.reduce_mean(self.predict_d_logits_for_g)
            tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')

            if self.FLAGS.dump_debug:
                tf.summary.image('D_predict_real', tf.transpose(tf.reshape(self.predict_d_logits,(-1,1,1,1)), (0, 2, 3, 1)), collections='D')
                tf.summary.image('D_predict_fake', tf.transpose(tf.reshape(self.predict_d_logits_for_g, (-1,1,1,1)), (0, 2, 3, 1)), collections='D')

        self.d_loss = self.d_loss_fake - self.d_loss_real
        tf.summary.scalar('d_loss', self.d_loss, collections='D')

        self.reg_loss_d = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='D')
        self.d_loss_no_reg = self.d_loss
        self.d_loss += self.reg_loss_d
        if self.FLAGS.dump_debug:
            tf.summary.scalar('d_loss_plus_reg', self.d_loss, collections='D')
            tf.summary.scalar('d_loss_reg_only', self.reg_loss_d, collections='D')

        # Generative loss
        # g_loss = tf.reduce_mean(ops.binary_cross_entropy(preds=self.predict_d_for_g, targets=tf.ones_like(self.predict_d_for_g)))
        g_loss = -tf.reduce_mean(self.predict_d_logits_for_g)

        tf.summary.scalar('g_loss', g_loss, collections='G2')

        # Context loss L2
        predict_image = tf.abs(tf.complex(real=self.predict_g2['real'], imag=self.predict_g2['imag']))
        label_image = tf.abs(tf.complex(real=self.labels['real'], imag=self.labels['imag']))
        self.context_loss = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(predict_image - label_image)))

        # self.context_loss = tf.reduce_mean(tf.square(real_diff) + tf.square(imag_diff), name='Context_loss_mean')
        print("You are using L2 loss")

        tf.summary.scalar('g_loss_context_only', self.context_loss, collections='G2')

        self.g_loss = self.adv_loss_w * g_loss + self.FLAGS.gen_loss_context * self.context_loss
        # self.g_loss = self.FLAGS.gen_loss_adversarial * g_loss + self.FLAGS.gen_loss_context * context_loss
        tf.summary.scalar('g_loss_plus_context', self.g_loss, collections='G2')

        # if len(self.regularization_values) > 0:
        # reg_loss_g = self.reg_w * tf.reduce_sum(self.regularization_values)
        self.reg_loss_g = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='G2')
        self.g_loss_no_reg = self.g_loss
        self.g_loss += self.reg_loss_g
        if self.FLAGS.dump_debug:
            tf.summary.scalar('g_loss_plus_context_plus_reg', self.g_loss, collections='G2')
            tf.summary.scalar('g_loss_reg_only', self.reg_loss_g, collections='D')

        tf.summary.scalar('diff-loss', tf.abs(self.d_loss - self.g_loss), collections='G2')

    def __clip_weights__(self):
        clip_ops = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if 'D_' in var.name:
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)))
        return tf.group(*clip_ops)

    def __training__(self, learning_rate):
        """
        :param learning_rate:
        :return:

        """
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D_' in var.name]
        self.g2_vars = [var for var in t_vars if 'G2_' in var.name]

        # Create RMSProb optimizer with the given learning rate.
        optimizer_d = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate, centered=True)
        optimizer_g2 = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate, centered=True)

        # Create a variable to track the global step.
        global_step_d = tf.Variable(0, name='global_step_d', trainable=False)
        global_step_g2 = tf.Variable(0, name='global_step_g2', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        grad_d = optimizer_d.compute_gradients(loss=self.d_loss, var_list=self.d_vars)
        grad_g = optimizer_g2.compute_gradients(loss=self.g_loss, var_list=self.g2_vars)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Ensures that we execute the update_ops before performing the train_step
        with tf.control_dependencies(self.update_ops):
            train_op_d = optimizer_d.apply_gradients(grad_d, global_step=global_step_d)
            train_op_g = optimizer_g2.apply_gradients(grad_g, global_step=global_step_g2)

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
