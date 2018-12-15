#!/usr/bin/env python
import tensorflow as tf
import time
from PIL import Image
import numpy as np

from model.image_utils import *


def conv_layer(input_x, out_channel, stride=2, kernel_shape=5, rand_seed=2813, index=0, prefix='d'):
    """
    :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param stride: the number for the sliding window.
    :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param index: The index of the layer. It is used for naming only.
    :param prefix: for distinguishing between generator and discriminator
    """

    with tf.variable_scope('{}_conv_layer_{}'.format(prefix, index)):
        with tf.name_scope('conv_kernel'):
            w_shape = [kernel_shape, kernel_shape, input_x.get_shape()[-1], out_channel]
            weight = tf.get_variable(name='{}_conv_kernel_{}'.format(prefix, index),
                                     shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('conv_bias'):
            b_shape = [out_channel]
            bias = tf.get_variable(name='{}_conv_bias_{}'.format(prefix, index), shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        conv_out = tf.nn.conv2d(input_x, weight, strides=[1, stride, stride, 1], padding="SAME")
        cell_out = tf.nn.relu(conv_out + bias)

        tf.summary.histogram('{}/conv_layer/{}/kernel'.format(prefix, index), weight)
        tf.summary.histogram('{}/conv_layer/{}/bias'.format(prefix, index), bias)
    return cell_out


def conv_transpose_layer(input_x, output_shape, stride=2, kernel_shape=5, rand_seed=2813, index=0, prefix='d'):
    """
    :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param outout_shape: A 1-D Tensor representing the output shape of the deconvolution op.
    :param stride: the number for the sliding window.
    :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param index: The index of the layer. It is used for naming only.
    :param prefix: for distinguishing between generator and discriminator
    """

    with tf.variable_scope('{}_conv_transpose_layer_{}'.format(prefix,index)):
        with tf.name_scope('conv_transpose_kernel'):
            w_shape = [kernel_shape, kernel_shape, output_shape[-1], input_x.get_shape()[-1]]
            weight = tf.get_variable(name='{}_conv_transpose_kernel_{}'.format(prefix, index),
                                     shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('conv_bias'):
            b_shape = [output_shape[-1]]
            bias = tf.get_variable(name='{}_conv_transpose_bias_{}'.format(prefix, index),
                                   shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        conv_out = tf.nn.conv2d_transpose(input_x, weight, output_shape=output_shape, strides=[1, stride, stride, 1])
        cell_out = tf.nn.bias_add(conv_out, bias)

        tf.summary.histogram('{}/conv_transpose_layer/{}/kernel'.format(prefix, index), weight)
        tf.summary.histogram('{}/conv_transpose_layer/{}/bias'.format(prefix, index), bias)

    return cell_out


def fc_layer(input_x, out_size, rand_seed, activation_function=None, index=0, prefix='d'):
    """
    :param input_x: The input of the FC layer. It should be a flatten vector.
    :param out_size: The length of output vector.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param activation_function: The activation function for the output. Default set to None.
    :param index: The index of the layer. It is used for naming only.
    :param prefix: for distinguishing between generator and discriminator
    """
    shape = input_x.get_shape().as_list()

    with tf.variable_scope('fc_layer_{}'.format(prefix, index)):
        with tf.name_scope('fc_kernel'):
            w_shape = [shape[1], out_size]
            weight = tf.get_variable(name='{}_fc_kernel_{}'.format(prefix, index),
                                     shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('fc_bias'):
            b_shape = [out_size]
            bias = tf.get_variable(name='{}_fc_bias_{}'.format(prefix, index),
                                   shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        cell_out = tf.add(tf.matmul(input_x, weight), bias)
        if activation_function is not None:
            cell_out = activation_function(cell_out)

        tf.summary.histogram('{}/fc_layer/{}/kernel'.format(prefix, index), weight)
        tf.summary.histogram('{}/fc_layer/{}/bias'.format(prefix, index), bias)

    return cell_out, weight, bias


def norm_layer(input_x, is_training):
    """
    :param input_x: The input that needed for normalization.
    :param is_training: To control the training or inference phase
    """
    with tf.variable_scope('batch_norm'):
        batch_mean, batch_variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def True_fn():
            ema_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        def False_fn():
            return ema.average(batch_mean), ema.average(batch_variance)

        mean, variance = tf.cond(is_training, True_fn, False_fn)

        cell_out = tf.nn.batch_normalization(input_x,
                                             mean,
                                             variance,
                                             offset=None,
                                             scale=None,
                                             variance_epsilon=1e-6,
                                             name=None)
    return cell_out


def train_step(loss, vars, prefix, learning_rate=1e-3):
    #loss = tf.constant(loss)
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate,
                name='{}_adam'.format(prefix)).minimize(loss, var_list=vars)
    return step


class DCGAN():
    def __init__(self, batch_size=64, z_dim=100, gf_dim=64, df_dim=64,
    model_name='DCGAN', data_source='Cars'):
        """
        :param batch_size: Size of each batch
        :param gf_dim: dimension of the filter generator for first convolution
        :param df_dim: dimension of the filter discriminator in first colvolution
        :param is_training: a boolean representing training / testing
        """

        self.batch_size = batch_size
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.z_dim = z_dim
        self.model_name = model_name
        self.seed = 92913
        self.data_source = data_source

        self.model()

    def generator(self, input_z):
        """
        """
        with tf.variable_scope("Generator"):
            # Proyect and reshape
            # Project - Fully Connected layer
            self.z_, self.h0_w, self.h0_b = fc_layer(input_z, out_size=self.gf_dim*8*4*4,
                                                        rand_seed=self.seed,
                                                        index=0,
                                                        prefix='g')

            # Reshape
            conv_layer_0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            print('conv_0_{}'.format(conv_layer_0.shape))
            # batch normalization
            conv_layer_0 = norm_layer(conv_layer_0, self.is_training)
            print('conv_0_norm_{}'.format(conv_layer_0.shape))
            # relu
            conv_layer_0 = tf.nn.relu(conv_layer_0)
            print('conv_0_relu_{}'.format(conv_layer_0.shape))

            # convolution 1
            conv_layer_1 = conv_transpose_layer(input_x=conv_layer_0,
                                                output_shape=[self.batch_size,
                                                    8, 8, self.gf_dim * 4],
                                                rand_seed=self.seed,
                                                index=1, prefix='g')
            print('conv_1_{}'.format(conv_layer_1.shape))
            # batch normalization
            conv_layer_1 = norm_layer(conv_layer_1, self.is_training)
            print('conv_1_norm_{}'.format(conv_layer_1.shape))
            # relu
            conv_layer_1 = tf.nn.relu(conv_layer_1)
            print('conv_1_relu_{}'.format(conv_layer_1.shape))

            # convolution 2
            conv_layer_2 = conv_transpose_layer(input_x=conv_layer_1,
                                                output_shape=[self.batch_size,
                                                    16, 16, self.gf_dim * 2],
                                                rand_seed=self.seed,
                                                index=2, prefix='g')
            print('conv_2_{}'.format(conv_layer_2.shape))
            # batch normalization
            conv_layer_2 = norm_layer(conv_layer_2, self.is_training)
            print('conv_2_norm_{}'.format(conv_layer_2.shape))
            # relu
            conv_layer_2 = tf.nn.relu(conv_layer_2)
            print('conv_2_relu_{}'.format(conv_layer_2.shape))

            # convolution 3
            conv_layer_3 = conv_transpose_layer(input_x=conv_layer_2,
                                                output_shape=[self.batch_size,
                                                    32, 32, self.gf_dim * 1],
                                                rand_seed=self.seed,
                                                index=3, prefix='g')
            print('conv_3_{}'.format(conv_layer_3.shape))
            # batch normalization
            conv_layer_3 = norm_layer(conv_layer_3, self.is_training)
            print('conv_3_norm_{}'.format(conv_layer_3.shape))
            # relu
            conv_layer_3 = tf.nn.relu(conv_layer_3)
            print('conv_3_relu_{}'.format(conv_layer_3.shape))

            # convolution 4
            conv_layer_4 = conv_transpose_layer(input_x=conv_layer_3,
                                                output_shape=[self.batch_size,
                                                    64, 64, 3],
                                                rand_seed=self.seed,
                                                index=4, prefix='g')
            print('conv_4_{}'.format(conv_layer_4.shape))
            # tanh
            return tf.nn.tanh(conv_layer_4, name='tanh_fake')

    def discriminator(self, image, reuse=False):
        """
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # convolution 0
        conv_layer_0 = conv_layer(image,
                                  out_channel=self.df_dim,
                                  rand_seed=self.seed,
                                  index=0, prefix='d')
        # leaky relu
        conv_layer_0 = tf.nn.leaky_relu(conv_layer_0, alpha=0.2)

        # convolution 1
        conv_layer_1 = conv_layer(conv_layer_0,
                                  out_channel=self.df_dim * 2,
                                  rand_seed=self.seed,
                                  index=1, prefix='d')
        # leaky relu
        conv_layer_1 = tf.nn.leaky_relu(conv_layer_1, alpha=0.2)

        # convolution 2
        conv_layer_2 = conv_layer(conv_layer_1,
                                  out_channel=self.df_dim * 4,
                                  rand_seed=self.seed,
                                  index=2, prefix='d')
        # leaky relu
        conv_layer_2 = tf.nn.leaky_relu(conv_layer_2, alpha=0.2)

        # convolution 3
        conv_layer_3 = conv_layer(conv_layer_2,
                                  out_channel=self.df_dim * 8,
                                  rand_seed=self.seed,
                                  index=3, prefix='d')
        # leaky relu
        conv_layer_3 = tf.nn.leaky_relu(conv_layer_3, alpha=0.2)

        # reshape
        norm_shape = conv_layer_3.get_shape()
        img_vector_length = norm_shape[1].value * norm_shape[2].value * norm_shape[3].value
        flatten = tf.reshape(conv_layer_3, shape=[-1, img_vector_length])
        # linear
        fc_layer_4, _, _ = fc_layer(flatten, out_size=1, rand_seed=self.seed,
                prefix='d', index=1)

        return tf.nn.sigmoid(fc_layer_4, name='sigmoid_real'), fc_layer_4

    def model(self):
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(shape=[self.batch_size, 64, 64, 3], dtype=tf.float32, name='real')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            self.z = tf.placeholder(shape=[self.batch_size, self.z_dim], dtype=tf.float32, name='z')
            # Generator
            self.fake_img = self.generator(self.z)

            print(self.img.shape)
            print(self.fake_img.shape)

            # Discriminator
            # with real images
            self.D, self.D_logits = self.discriminator(self.img)
            # with fake images
            self.D_, self.D_logits_ = self.discriminator(self.fake_img, reuse=True)

            #with tf.name_scope("loss"):
            # real image loss for discriminator
            d_real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                            labels=tf.ones_like(self.D)),
                    name='discriminator_loss_real_images')
            # fake image loss for discriminator
            d_fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                            labels=tf.zeros_like(self.D_)),
                    name='discriminator_loss_fake_images')
            # Discriminator loss:
            self.d_loss = tf.add(d_real_loss, d_fake_loss, name='discriminator_loss')
            tf.summary.scalar('Discriminator_loss', self.d_loss)

            # Generator loss:
            self.g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                            labels=tf.ones_like(self.D_)),
                    name='generator_loss')
            tf.summary.scalar('Generator_loss', self.g_loss)

            # Get all variables and split them for each CNN:
            t_vars = tf.trainable_variables()
            # Generator variables
            self.g_vars = [var for var in t_vars if 'g_' in var.name]
            # Discriminator variables
            self.d_vars = [var for var in t_vars if 'd_' in var.name]

    def train(self, X_train, learning_rate=0.0001, iters=1500):
        print("Building my DCGAN")

        img_gen = ImageCollector(X_train)
        # Generate augmentation
        new_train_size = img_gen.x_aug.shape[0]
        print('X_train size {}'.format(X_train.shape[0]))
        print('new size {}'.format(new_train_size))

        # calculate loss
        #print(self.g_loss)
        #print(self.g_vars)
        g_step = train_step(self.g_loss, self.g_vars, 'g', learning_rate)
        d_step = train_step(self.d_loss, self.d_vars, 'd', learning_rate)

        # Open session
        self.session = tf.Session()
        with self.session as sess:
            # Initialize z
            input_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
            input_z = input_z.astype(np.float32)

            merge = tf.summary.merge_all()
            saver = tf.train.Saver()
            cur_model_name = 'my_dcgan_{}'.format(int(time.time()))
            sess.run(tf.global_variables_initializer())

            batch_gen = img_gen.next_batch_gen(batch_size=self.batch_size, shuffle=True)

            for itr in range(iters):
                img_batch = next(batch_gen)
                # Update Generator
                _, g_loss = sess.run([g_step, self.g_loss], feed_dict={self.img: img_batch,
                                                   self.z: input_z,
                                                   self.is_training: True})
                # Update Discriminator
                _, d_loss = sess.run([d_step, self.d_loss],
                        feed_dict={self.img: img_batch,
                                                   self.z: input_z,
                                                   self.is_training: True})
                if itr % 100 == 0:
                    [d_loss, g_loss, fake_img] = sess.run([self.d_loss,
                        self.g_loss, self.fake_img], feed_dict={self.img: img_batch,
                                                                self.z: input_z,
                                                                self.is_training: True})
                    print("Step: {}, D_loss: {}, G_loss: {}".format(itr, d_loss, g_loss))
                    # Store generated fake image
                    Image.fromarray(np.uint8((fake_img[0, :, :, :] + 1.0) *
                        127.5)).save("../{}/results/{}.jpg".format(self.data_source, itr))

                # Store checkpoint
                if itr % 200 == 0:
                    saver.save(sess,
                            "../{}/checkpoints/{}.ckpt".format(self.data_source, self.model_name, itr))

