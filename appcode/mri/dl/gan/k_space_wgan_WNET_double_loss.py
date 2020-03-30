import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops
import pdb

class KspaceWgan(BasicModel):
    """
    Represents k-space super resolution model
    """

    def __init__(self,real,imag,input=None, dims_in=None, dims_out=None, FLAGS=None, train_phase=None, adv_loss_w=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=input, labels=input, dims_in=dims_in, dims_out=dims_out)
        self.input = input
        if real is not None:
            self.input['real'] = real
            self.input['imag'] = imag

        print("Input is y label")

        self.input_d={}
        real_input = self.input['real'][:,1,:,:]
        real_input = tf.reshape(real_input,(-1,1,self.dims_in[1],self.dims_in[2]))
        self.input_d['real'] = real_input
        imag_input = self.input['imag'][:,1,:,:]
        imag_input = tf.reshape(imag_input,(-1,1,self.dims_in[1],self.dims_in[2]))
        self.input_d['imag'] = imag_input

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
        # self.reg_w = self.FLAGS.regularization_weight
        self.regularization_values = []
        self.regularization_sum = None

        self.regularization_values_d = []
        self.regularization_sum_d = None

        self.clip_weights = None

        self.real_im = None
        self.fake_im = None

        tf.get_collection('D')
        tf.get_collection('G')

    def build(self):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g,self.predict_g_kspace = self.__G__()

        with tf.name_scope('D_'):
            self.predict_logits, self.predict = self.__D__(self.input_d, self.predict_g, input_type="Real")

            self.predict_d, self.predict_d_for_g = tf.split(value=self.predict, num_or_size_splits=2, axis=0)
            self.predict_d_logits, self.predict_d_logits_for_g = tf.split(value=self.predict_logits, num_or_size_splits=2, axis=0)

            # self.predict_d, self.predict_d_logits
            # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            #     self.predict_d_for_g, self.predict_d_logits_for_g = self.__D__(self.predict_g, input_type="Gen")

            if len(self.regularization_values_d) > 0:
                self.regularization_sum_d = sum(self.regularization_values_d)

            self.clip_weights = self.__clip_weights__()

        with tf.name_scope('loss'):
            # self.loss_g = self.__loss_g__(predict=self.predict_g, self.labels, reg=self.regularization_sum)
            self.__loss__()

        with tf.name_scope('training'):
            self.train_op_d, self.train_op_g , self.train_op_u, self.train_op_k = self.__training__(learning_rate=self.FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy L2 norm
            self.evaluation = self.__evaluation__(predict=self.predict_g, labels=self.labels)

    def __G__(self):
        """
        Define the model
        """

        # Create the inputs

        mask = tf.concat([self.input['mask_1'],self.input['mask_2'],self.input['mask_3']],1)
        mask_not = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)
        x_real = self.input['real'] * mask
        x_imag = self.input['imag'] * mask

        # print(self.input['real'][:, 0, :, :])
        # # Output to Tensorboard:
        # input_real = tf.concat(axis=0, values=[self.input['real'][:,0,:,:],x_real[:,0,:,:]])
        # input_imag = tf.concat(axis=0, values=[self.input['imag'][:,0,:,:],x_imag[:,0,:,:]])
        #
        # # Input d holds real&imaginary values. The discriminative decision based on reconstructed image
        # input_to_discriminator = self.get_reconstructed_image(real=input_real, imag=input_imag, name='Both')
        #
        # org, fake = tf.split(input_to_discriminator, num_or_size_splits=2, axis=0)
        #
        # org = tf.reshape(tf.abs(tf.complex(real=tf.squeeze(org[:, 0, :, :]), imag=tf.squeeze(org[:, 1, :, :]))),
        #                  shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        # fake = tf.reshape(tf.abs(tf.complex(real=tf.squeeze(fake[:, 0, :, :]), imag=tf.squeeze(fake[:, 1, :, :]))),
        #                   shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        #
        # tf.summary.image('G_input_slice0', tf.transpose(org, (0, 2, 3, 1)), collections='Input',
        #                  max_outputs=4)
        # tf.summary.image('G_masked_input_slice0', tf.transpose(fake, (0, 2, 3, 1)), collections='Input',
        #                  max_outputs=4)

        print "Noise level: (-0.01,0.01)"
        minval = -0.01
        maxval = 0.01

        noise_real = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval, dtype=tf.float32, seed=None, name='z_real')
        noise_imag = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval, dtype=tf.float32, seed=None, name='z_imag')
        x_real,r_mean,r_std = normalize(x_real)
        x_imag,i_mean,i_std = normalize(x_imag)

        x_real += noise_real
        x_imag += noise_imag

        if self.FLAGS.dump_debug:
            tf.summary.image('G_mask', tf.transpose(self.labels['mask_2'], (0, 2, 3, 1)), collections='G', max_outputs=1)
            tf.summary.image('noise_real', tf.transpose(noise_real, (0, 2, 3, 1)), collections='G', max_outputs=1)
            tf.summary.image('noise_image', tf.transpose(noise_imag, (0, 2, 3, 1)), collections='G', max_outputs=1)
            tf.summary.image('x_real_noise', tf.transpose(x_real, (0, 2, 3, 1)), collections='G', max_outputs=2)
            tf.summary.image('x_image_noise', tf.transpose(x_imag, (0, 2, 3, 1)), collections='G', max_outputs=2)
            tf.summary.image('x_input_real', tf.transpose(self.input['real'], (0, 2, 3, 1)), collections='G', max_outputs=2)
            tf.summary.image('x_input_image', tf.transpose(self.input['imag'], (0, 2, 3, 1)), collections='G', max_outputs=2)

            image_with_noise_padding = self.get_reconstructed_image(real=x_real, imag=x_imag, name='NoisePadding')
            image_with_zero_padding = self.get_reconstructed_image(real=self.input['real'] * self.input['mask_2'],
                                                                    imag=self.input['imag'] * self.input['mask_2'], name='NoisePadding')
            image_debug = self.get_reconstructed_image(real=self.input['real'],
                                                                    imag=self.input['imag'], name='RegularDebug')
            image_with_noise_padding = tf.expand_dims(input=tf.abs(tf.complex(real=image_with_noise_padding[:,0,:,:],
                                                                              imag=image_with_noise_padding[:,1,:,:])), dim=1)
            image_with_zero_padding = tf.expand_dims(input=tf.abs(tf.complex(real=image_with_zero_padding[:,0,:,:],
                                                                             imag=image_with_zero_padding[:,1,:,:])), dim=1)
            image_debug = tf.expand_dims(input=tf.abs(tf.complex(real=image_debug[:,0,:,:],
                                                                              imag=image_debug[:,1,:,:])), dim=1)
            tf.summary.image('image_noise_padding', tf.transpose(image_with_noise_padding, (0, 2, 3, 1)), collections='G', max_outputs=2)
            tf.summary.image('image_zero_padding', tf.transpose(image_with_zero_padding, (0, 2, 3, 1)), collections='G', max_outputs=2)
            tf.summary.image('image_zero_padding', tf.transpose(image_debug, (0, 2, 3, 1)), collections='G', max_outputs=2)

        self.x_input_upscale['real'] = x_real
        self.x_input_upscale['imag'] = x_imag
        # Model convolutions
        # with tf.name_scope('real'):
        #x_input_stack = tf.concat([x_real[:,:,:,:], x_imag[:,:,:,:],self.input['X_loc'],self.input['Y_loc']], axis=1)
        x_input_stack = tf.concat([x_real[:, :, :, :], x_imag[:, :, :, :]],axis=1)

        # UNET Model convolutions
        out_dim = 64  # 256x256 => 128x128x64
        conv1 = ops.conv2d(x_input_stack, output_dim=out_dim, k_h=4, k_w=4, d_h=2, d_w=2, name="G_K_conv_1")

        out_dim = 128  # 256x256 => 128x128 /2
        conv2 = ops.UConv(conv1, n_filters=out_dim, training=self.train_phase, name='G_K_conv2')

        out_dim = 256  # 128x128 => 64x64 /2
        conv3 = ops.UConv(conv2, n_filters=out_dim, training=self.train_phase, name='G_K_conv3')

        out_dim = 512  # 64x64 => 32x32 /2
        conv4 = ops.UConv(conv3, n_filters=out_dim, training=self.train_phase, name='G_K_conv4')

        out_dim = 512  # 32x32 => 16x16
        conv5 = ops.UConv(conv4, n_filters=out_dim, training=self.train_phase, name='G_K_conv5')

        up4 = conv5
        out_dim = 1024
        up4 = tf.layers.conv2d_transpose(up4, out_dim, [4, 4], [2, 2], padding='same', activation=None,
                                         data_format="channels_first", name='G_K_conv4')
        up4 = tf.nn.relu(tf.layers.batch_normalization(up4, training=self.train_phase, axis=1, name='G_K_conv4_bn'),
                         name='G_K_conv4_act')

        up3 = tf.concat([up4, conv4], axis=1, name="concat_3k")
        out_dim = 256
        up3 = tf.layers.conv2d_transpose(up3, out_dim, [4, 4], [2, 2], padding='same', activation=None,
                                         data_format="channels_first", name='G_K_conv5')
        up3 = tf.nn.relu(tf.layers.batch_normalization(up3, training=self.train_phase, axis=1, name='G_K_conv5_bn'),
                         name='G_K_conv5_act')

        up2 = tf.concat([up3, conv3], axis=1, name="concat_2k")
        out_dim = 128
        up2 = tf.layers.conv2d_transpose(up2, out_dim, [4, 4], [2, 2], padding='same', activation=None,
                                         data_format="channels_first", name='G_K_conv6')
        up2 = tf.nn.relu(tf.layers.batch_normalization(up2, training=self.train_phase, axis=1, name='G_K_conv6_bn'),
                         name='G_K_conv6_act')

        up1 = tf.concat([up2, conv2], axis=1, name="concat_1k")
        out_dim = 64
        up1 = tf.layers.conv2d_transpose(up1, out_dim, [4, 4], [2, 2], padding='same', activation=None,
                                         data_format="channels_first", name='G_K_conv7')
        up1 = tf.nn.relu(tf.layers.batch_normalization(up1, training=self.train_phase, axis=1, name='G_K_conv7_bn'),
                         name='G_K_conv7_act')

        up0 = tf.concat([up1, conv1], axis=1, name="concat_0k")
        out_dim = 64
        up0 = tf.layers.conv2d_transpose(up0, out_dim, [4, 4], [2, 2], padding='same', activation=None,
                                         data_format="channels_first", name='G_K_conv8')
        up0 = tf.nn.relu(tf.layers.batch_normalization(up0, training=self.train_phase, axis=1, name='G_K_conv8_bn'),
                         name='G_K_conv8_act')

        out_dim = 2
        self.conv_last_kspace = ops.conv2d(up0, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_K_conv_last")

        #Denorm


        predict_kspace = {}
        predict_kspace['real'] = tf.reshape(self.conv_last_kspace[:,0,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_real')
        predict_kspace['imag'] = tf.reshape(self.conv_last_kspace[:,1,:,:], [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='G_predict_imag')

        predict_kspace['real'] = denormalize(predict_kspace['real'],r_mean,r_std)
        predict_kspace['imag'] = denormalize(predict_kspace['imag'], i_mean, i_std)
        # Masking
        predict_kspace['real'] = tf.multiply(predict_kspace['real'], mask_not[:,1,:,:])
        predict_kspace['imag'] = tf.multiply(predict_kspace['imag'], mask_not[:,1,:,:])
        input_masked_real = tf.multiply(self.input_d['real'], self.labels['mask_2'], name='input_masked_real')
        input_masked_imag = tf.multiply(self.input_d['imag'], self.labels['mask_2'], name='input_masked_imag')

        with tf.name_scope("middle_predict"):
            predict_kspace['real'] = tf.add(predict_kspace['real'], input_masked_real, name='real')
            predict_kspace['imag'] = tf.add(predict_kspace['imag'], input_masked_imag, name='imag')

        tf.add_to_collection("predict", predict_kspace['real'])
        tf.add_to_collection("predict", predict_kspace['imag'])

        # Dump prediction out
        if self.FLAGS.dump_debug:
            tf.summary.image('G_predict_real', tf.transpose(predict_kspace['real'], (0, 2, 3, 1)), collections='G')
            tf.summary.image('G_predict_imag', tf.transpose(predict_kspace['imag'], (0, 2, 3, 1)), collections='G')

        #Get image space:
        image_space = self.get_reconstructed_image(real=predict_kspace['real'], imag=predict_kspace['imag'],name="predict_g1")
        image_space = tf.abs(tf.complex(real=tf.expand_dims(image_space[:, 0, :, :], axis=1),
                          imag=tf.expand_dims(image_space[:, 1, :, :], axis=1)))

        image_space = (image_space - tf.reduce_min(image_space))/(tf.reduce_max(image_space)-tf.reduce_min(image_space))*2 - 1
        tf.summary.image('G_First_predict', tf.transpose(image_space, (0, 2, 3, 1)), collections='G',max_outputs=4)
        predict_kspace['image'] = image_space
        #Inits
        # UNET Model convolutions
        out_dim = 64  # 256x256 => 128x128x64
        conv1 = ops.conv2d(image_space, output_dim=out_dim, k_h=4, k_w=4, d_h=2, d_w=2, name="G_U_conv_1")

        out_dim = 128  # 256x256 => 128x128 /2
        conv2 = ops.UConv(conv1, n_filters=out_dim,training=self.train_phase, name='G_U_conv2')

        out_dim = 256  # 128x128 => 64x64 /2
        conv3 = ops.UConv(conv2, n_filters=out_dim,training=self.train_phase, name='G_U_conv3')

        out_dim = 512  # 64x64 => 32x32 /2
        conv4 = ops.UConv(conv3, n_filters=out_dim,training=self.train_phase, name='G_U_conv4')

        out_dim = 512  # 32x32 => 16x16
        conv5 = ops.UConv(conv4, n_filters=out_dim,training=self.train_phase, name='G_U_conv5')

        # out_dim = 512  # 16x16 => 8x8
        # conv6 = ops.UConv(conv5, n_filters=out_dim,training=self.train_phase, name='G_U_conv6')
        #
        # out_dim = 512  # 8x8 => 4x4
        # conv7 = ops.UConv(conv6, n_filters=out_dim,training=self.train_phase, name='G_U_conv7')
        #
        # out_dim = 512  # 2x2 => 1x1
        # conv8 = ops.UConv(conv7, n_filters=out_dim,training=self.train_phase, name='G_U_conv8')
        #
        # out_dim = 512  # 1x1 => 2x2
        # up7 = tf.layers.conv2d_transpose(conv8,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_Up_conv1')
        # up7 = tf.nn.relu(tf.layers.batch_normalization(up7,training=self.train_phase,axis=1,name='G_U_conv1_bn'),name='G_U_conv1_act')
        #
        # up6 = tf.concat([up7, conv7], axis=1, name="concat_6")
        # out_dim = 1024
        # up6 = tf.layers.conv2d_transpose(up6,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv2')
        # up6 = tf.nn.relu(tf.layers.batch_normalization(up6,training=self.train_phase,axis=1,name='G_U_conv2_bn'),name='G_U_conv2_act')
        #
        # up5 = tf.concat([up6, conv6], axis=1, name="concat_5")
        # out_dim = 1024
        # up5 = tf.layers.conv2d_transpose(up5,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv3')
        # up5 = tf.nn.relu(tf.layers.batch_normalization(up5,training=self.train_phase,axis=1,name='G_U_conv3_bn'),name='G_U_conv3_act')

        # up4 = tf.concat([up5, conv5], axis=1, name="concat_4")
        up4 = conv5
        out_dim = 1024
        up4 = tf.layers.conv2d_transpose(up4,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv4')
        up4= tf.nn.relu(tf.layers.batch_normalization(up4,training=self.train_phase,axis=1,name='G_U_conv4_bn'),name='G_U_conv4_act')

        up3 = tf.concat([up4, conv4], axis=1, name="concat_3")
        out_dim = 256
        up3 = tf.layers.conv2d_transpose(up3,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv5')
        up3= tf.nn.relu(tf.layers.batch_normalization(up3,training=self.train_phase,axis=1,name='G_U_conv5_bn'),name='G_U_conv5_act')

        up2 = tf.concat([up3, conv3], axis=1, name="concat_2")
        out_dim = 128
        up2 = tf.layers.conv2d_transpose(up2,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv6')
        up2 = tf.nn.relu(tf.layers.batch_normalization(up2,training=self.train_phase,axis=1,name='G_U_conv6_bn'),name='G_U_conv6_act')

        up1 = tf.concat([up2, conv2], axis=1, name="concat_1")
        out_dim = 64
        up1 = tf.layers.conv2d_transpose(up1,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv7')
        up1= tf.nn.relu(tf.layers.batch_normalization(up1,training=self.train_phase,axis=1,name='G_U_conv7_bn'),name='G_U_conv7_act')

        up0 = tf.concat([up1, conv1], axis=1, name="concat_0")
        out_dim = 64
        up0 = tf.layers.conv2d_transpose(up0,out_dim,[4,4],[2,2],padding='same',activation=None,data_format="channels_first",name='G_U_conv8')
        up0= tf.nn.relu(tf.layers.batch_normalization(up0,training=self.train_phase,axis=1,name='G_U_conv8_bn'),name='G_U_conv8_act')

        out_dim = 1
        conv_last = ops.conv2d(up0, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="G_U_conv_last")
        conv_last = tf.nn.tanh(conv_last, name="G_U_act_last")


        predict = {}

        with tf.name_scope("final_predict"):
            pred_image = conv_last + image_space
            pred_image = tf.clip_by_value(pred_image,clip_value_min=-1,clip_value_max=1)
            # pred_image = (pred_image - tf.reduce_min(pred_image))/(tf.reduce_max(pred_image)-tf.reduce_min(pred_image))

        predict['image'] = pred_image
        tf.add_to_collection("predict_final", predict['image'])

        # Dump prediction out
        tf.summary.image('G_predict_diff',tf.transpose(pred_image-image_space, (0,2,3,1)), collections='G', max_outputs=4)
        tf.summary.image('G_predict_final',tf.transpose(predict['image'], (0,2,3,1)), collections='G', max_outputs=4)

        kspace_final = self.get_kspace_image(pred_image)
        real_final = tf.expand_dims(kspace_final[:, 0, :, :], axis=1)
        imag_final = tf.expand_dims(kspace_final[:, 1, :, :], axis=1)
        predict['real'] = real_final
        predict['imag'] = imag_final
        return predict , predict_kspace

    def __D__(self, input_d,predict_g, input_type):
        """
        Define the discriminator
        """
        # Dump input image out
        input_real = input_d['real']
        input_imag = input_d['imag']

        # Input d holds real&imaginary values. The discriminative decision based on reconstructed image

        fake = predict_g['image']

        org = self.get_reconstructed_image(real=input_real, imag=input_imag,name='input_d')

        org = tf.reshape(tf.abs(tf.complex(real=tf.squeeze(org[:,0,:,:]), imag=tf.squeeze(org[:,1,:,:]))), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        org = (org - tf.reduce_min(org)) / (tf.reduce_max(org) - tf.reduce_min(org))*2 -1
        org_input = tf.concat([org,fake],axis=1)
        fake_input = tf.concat([fake, org], axis=1)
        input_to_discriminator = tf.concat([org_input,fake_input],axis=0)
        # input_to_discriminator = tf.concat([org,fake],axis=0)
        self.real_im = org
        self.fake_im = fake
        kspace_org_n = self.get_kspace_image(org)
        self.real_org_n = tf.expand_dims(kspace_org_n[:, 0, :, :], axis=1)
        self.image_org_n = tf.expand_dims(kspace_org_n[:, 1, :, :], axis=1)

        tf.summary.image('D_x_input_reconstructed' + 'Original', tf.transpose(org, (0,2,3,1)), collections='D', max_outputs=4)
        tf.summary.image('D_x_input_reconstructed' + 'Fake', tf.transpose(fake, (0,2,3,1)), collections='G', max_outputs=4)

        # Model convolutions
        out_dim = 64  # 256X256
        net = ops.conv2d(input_to_discriminator, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="D_conv_1")
        net = ops.lrelu(net)

        out_dim = 64  # 128X128
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=2, d_w=2,
                                            name="D_conv_2")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn2")

        out_dim = 128  # 128X128
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                              name="D_conv_3")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn3")

        out_dim = 128  # 64X64
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=2, d_w=2,
                              name="D_conv_4")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn4")

        out_dim = 256  # 64X64
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                         name="D_conv_5")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn5")

        out_dim = 256  # 32X32
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=2, d_w=2,
                         name="D_conv_6")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn6")

        out_dim = 512  # 16X16
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1,
                         name="D_conv_7")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn7")

        out_dim = 512  # 8X8
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=2, d_w=2,
                         name="D_conv_8")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn8")

        out_dim = 512  # 4X4
        net = ops.conv2d(net, output_dim=out_dim, k_h=3, k_w=3, d_h=2, d_w=2,
                         name="D_conv_9")
        net = ops.lrelu(net)
        net = ops.batch_norm(net, self.train_phase, name="D_bn9")

        out_dim = 1024
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,out_dim,tf.nn.leaky_relu,name="D_dense_1")
        out_dim = 1
        predict_d = tf.layers.dense(net,out_dim,tf.nn.leaky_relu,name="D_dense_2")

        # Dump prediction out

        return  predict_d, tf.nn.sigmoid(predict_d)


    def __loss__(self):
        """
        Calculate loss
        :return:
        """

        first_predict = self.predict_g_kspace['image']
        with tf.variable_scope("discriminator") as scope:
            self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.predict_d_logits)+tf.random_normal(tf.shape(self.predict_d_logits),0,0.1),self.predict_d_logits)
            tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')
            # scope.reuse_variables()
            self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.predict_d_logits_for_g),self.predict_d_logits_for_g)
            tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')
            self.d_loss = self.d_loss_fake + self.d_loss_real

            # self.d_loss_real = tf.reduce_mean(self.predict_d_logits)
            # tf.summary.scalar('d_loss_real', self.d_loss_real, collections='D')
            # # scope.reuse_variables()
            # self.d_loss_fake = tf.reduce_mean(self.predict_d_logits_for_g)
            # tf.summary.scalar('d_loss_fake', self.d_loss_fake, collections='D')
            # self.d_loss = self.d_loss_fake - self.d_loss_real

            if self.FLAGS.dump_debug:
                tf.summary.image('D_predict_real', tf.transpose(tf.reshape(self.predict_d_logits,(-1,1,1,1)), (0, 2, 3, 1)), collections='D')
                tf.summary.image('D_predict_fake', tf.transpose(tf.reshape(self.predict_d_logits_for_g, (-1,1,1,1)), (0, 2, 3, 1)), collections='D')

        tf.summary.scalar('d_loss', self.d_loss, collections='D')

        # if len(self.regularization_values_d) > 0:
        # reg_loss_d = self.reg_w * tf.reduce_sum(self.regularization_values_d)
        self.reg_loss_d = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='D')
        self.d_loss_no_reg = self.d_loss
        self.d_loss += self.reg_loss_d
        if self.FLAGS.dump_debug:
            tf.summary.scalar('d_loss_plus_reg', self.d_loss, collections='D')
            tf.summary.scalar('d_loss_reg_only', self.reg_loss_d, collections='D')

        # Generative loss
        g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.predict_d_logits_for_g),self.predict_d_logits_for_g)
        # g_loss = -tf.reduce_mean(self.predict_d_logits_for_g)

        tf.summary.scalar('g_loss', g_loss, collections='G')


        # Context loss L2
        mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask_2'], tf.bool)), tf.float32)
        real_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g_kspace['real'] - self.input_d['real'], mask_not))
        imag_diff = tf.contrib.layers.flatten(tf.multiply(self.predict_g_kspace['imag'] - self.input_d['imag'], mask_not))
        self.context_loss = tf.reduce_mean(tf.square(real_diff) + tf.square(imag_diff), name='Context_loss_mean')

        self.final_context_loss = tf.reduce_mean(tf.square(self.real_org_n-self.predict_g['real']) + tf.square(self.image_org_n-self.predict_g['imag']), name='Context_loss_mean_final')


        tf.summary.scalar('first_kspace_loss', self.context_loss, collections='G')
        tf.summary.scalar('final_kspace_loss', self.final_context_loss, collections='G')

        self.g_loss = self.adv_loss_w * g_loss
        # self.g_loss = self.FLAGS.gen_loss_adversarial * g_loss + self.FLAGS.gen_loss_context * context_loss
        tf.summary.scalar('g_loss_plus_context', self.g_loss, collections='G')

        # if len(self.regularization_values) > 0:
        # reg_loss_g = self.reg_w * tf.reduce_sum(self.regularization_values)

        #Context loss Image space after unet
        g_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(self.fake_im,self.real_im), axis=[1, 2, 3]))
        g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(self.real_im), axis=[1, 2, 3]))
        self.im_loss = tf.reduce_mean(g_nmse_a / g_nmse_b)
        tf.summary.scalar('final_im_l2_loss', self.im_loss, collections='G')
        g_nl1_a = tf.sqrt(tf.reduce_sum(tf.abs(self.fake_im- self.real_im), axis=[1, 2, 3]))
        g_nl1_b = tf.sqrt(tf.reduce_sum(tf.abs(self.real_im), axis=[1, 2, 3]))
        self.l1_loss = tf.reduce_mean(g_nl1_a / g_nl1_b)

        g_k_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(first_predict, self.real_im), axis=[1, 2, 3]))

        self.first_im_loss = tf.reduce_mean(g_k_nmse_a/g_nmse_b)
        tf.summary.scalar('first_im_loss', self.first_im_loss, collections='G')

        self.context_loss_total = self.FLAGS.final_gen_loss_context* self.final_context_loss + self.FLAGS.gen_loss_context * self.context_loss + self.FLAGS.im_loss_context * self.im_loss + self.FLAGS.im_loss_context_l1 *self.l1_loss + self.FLAGS.first_im_loss_context * self.first_im_loss

        self.g_loss += self.context_loss_total

        self.reg_loss_g = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='G')
        self.g_loss_no_reg = self.g_loss
        self.g_loss += self.reg_loss_g

        if self.FLAGS.dump_debug:
            tf.summary.scalar('g_loss_plus_context_plus_reg', self.g_loss, collections='G')
            tf.summary.scalar('g_loss_reg_only', self.reg_loss_g, collections='D')

        tf.summary.scalar('diff-loss', tf.abs(self.d_loss - self.g_loss), collections='G')
        #PSNR
        range_diff = tf.reduce_max(self.real_im,axis=[1,2,3]) - tf.reduce_min(self.real_im,axis=[1,2,3])
        psnr_mse = tf.reduce_mean(tf.squared_difference(self.fake_im,self.real_im), axis=[1, 2, 3])
        psnr_mid = (range_diff**2)/psnr_mse
        self.psnr = tf.reduce_mean(10*tf.log(psnr_mid)/tf.log(tf.constant(10, dtype=psnr_mid.dtype)))
        tf.summary.scalar('g_loss_PSNR', self.psnr, collections='G')


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
        self.g_vars = [var for var in t_vars if 'G_' in var.name]
        self.u_vars = [var for var in t_vars if 'G_' in var.name]
        self.k_vars = [var for var in t_vars if 'K_' in var.name]

        # Create RMSProb optimizer with the given learning rate.
        optimizer_d = tf.train.AdamOptimizer(self.FLAGS.learning_rate,beta1=0.5)
        optimizer_g = tf.train.AdamOptimizer(self.FLAGS.learning_rate,beta1=0.5)
        optimizer_u = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        optimizer_k = tf.train.AdamOptimizer(self.FLAGS.learning_rate)

        # Create a variable to track the global step.
        global_step_d = tf.Variable(0, name='global_step_d', trainable=False)
        global_step_g = tf.Variable(0, name='global_step_g', trainable=False)
        global_step_u = tf.Variable(0, name='global_step_u', trainable=False)
        global_step_k = tf.Variable(0, name='global_step_k', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        grad_d = optimizer_d.compute_gradients(loss=self.d_loss, var_list=self.d_vars)
        grad_g = optimizer_g.compute_gradients(loss=self.g_loss, var_list=self.g_vars)
        grad_u = optimizer_u.compute_gradients(loss=self.context_loss_total, var_list=self.u_vars)
        grad_k = optimizer_k.compute_gradients(loss=self.context_loss, var_list=self.k_vars)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Ensures that we execute the update_ops before performing the train_step
        with tf.control_dependencies(self.update_ops):
            train_op_d = optimizer_d.apply_gradients(grad_d, global_step=global_step_d)
            train_op_g = optimizer_g.apply_gradients(grad_g, global_step=global_step_g)
            train_op_u = optimizer_u.apply_gradients(grad_u, global_step=global_step_u)
            train_op_k = optimizer_u.apply_gradients(grad_k, global_step=global_step_k)



        return train_op_d, train_op_g ,train_op_u , train_op_k

    def __evaluation__(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """
        # evalu = tf.reduce_mean(tf.square(tf.squeeze(predict['real']) - tf.squeeze(labels['real']))) \
        #         + tf.reduce_mean(tf.square(tf.squeeze(predict['imag']) - tf.squeeze(labels['imag'])))
        evalu = {'k_space':self.context_loss,"PSNR":self.psnr,"NMSE":self.im_loss}
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

    def get_kspace_image(self, image, name=None):
        """
        :param real:
        :param imag:
        :param name:
        :return:
        """
        kspace_image_complex = tf.expand_dims(tf.fft2d(tf.cast(image,tf.complex64)), axis=1)

        kspace_real = tf.reshape(tf.real(kspace_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        kspace_imag = tf.reshape(tf.imag(kspace_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])

        # Shifting
        top, bottom = tf.split(kspace_real, num_or_size_splits=2, axis=2)
        top_left, top_right = tf.split(top, num_or_size_splits=2, axis=3)
        bottom_left, bottom_right = tf.split(bottom, num_or_size_splits=2, axis=3)

        top_shift = tf.concat(axis=3, values=[bottom_right, bottom_left])
        bottom_shift = tf.concat(axis=3, values=[top_right, top_left])
        shifted_image = tf.concat(axis=2, values=[top_shift, bottom_shift])

        # Shifting
        top_imag, bottom_imag = tf.split(kspace_imag, num_or_size_splits=2, axis=2)
        top_left_imag, top_right_imag = tf.split(top_imag, num_or_size_splits=2, axis=3)
        bottom_left_imag, bottom_right_imag = tf.split(bottom_imag, num_or_size_splits=2, axis=3)

        top_shift_imag = tf.concat(axis=3, values=[bottom_right_imag, bottom_left_imag])
        bottom_shift_imag = tf.concat(axis=3, values=[top_right_imag, top_left_imag])
        shifted_image_imag = tf.concat(axis=2, values=[top_shift_imag, bottom_shift_imag])

        shifted_image_two_channels = tf.stack([shifted_image[:, 0, :, :], shifted_image_imag[:, 0, :, :]], axis=1)
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

        return self.FLAGS.reg_w * (reg_w )  + self.FLAGS.reg_b * reg_b

def normalize(tensor):
    mean, var = tf.nn.moments(tensor, axes=[1,2,3,0])
    std = tf.sqrt(var)
    norm_tensor = (tensor-mean)/std
    return norm_tensor,mean,std

def denormalize(tensor,mean,std):
    tensor = tensor*std
    tensor = tensor+mean
    return tensor