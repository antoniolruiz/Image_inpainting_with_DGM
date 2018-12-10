#!/usr/bin/env python
import tensorflow as tf


def conv_layer(input_x, out_channel, stride=2, kernel_shape=5, rand_seed=2813, index=0):
    """
    :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param index: The index of the layer. It is used for naming only.
    """

    with tf.variable_scope('conv_layer_%d' % index):
        with tf.name_scope('conv_kernel'):
            w_shape = [kernel_shape, kernel_shape, input_.get_shape()[-1], out_channel]
            weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('conv_bias'):
            b_shape = [out_channel]
            bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        conv_out = tf.nn.conv2d(input_x, weight, strides=[1, stride, stride, 1], padding="SAME")
        cell_out = tf.nn.relu(conv_out + bias)

        tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
        tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)
    return cell_out



def conv_transpose_layer(input_x, output_shape, stride=2, kernel_shape=5, rand_seed=2813, index=0):
    """
    :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param index: The index of the layer. It is used for naming only.
    """

    with tf.variable_scope('conv_transpose_layer_%d' % index):
        with tf.name_scope('conv_transpose_kernel'):
            w_shape = [kernel_shape, kernel_shape, output_shape[-1], input_x.get_shape()[-1]]
            weight = tf.get_variable(name='conv_transpose_kernel_%d' % index, shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('conv_bias'):
            b_shape = [output_shape[-1]]
            bias = tf.get_variable(name='conv_transpose_bias_%d' % index, shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        conv_out = tf.nn.conv2d_transpose(input_x, weight, output_shape=output_shape, strides=[1, stride, stride, 1])
        cell_out = tf.nn.bias_add(conv_out, bias)

        tf.summary.histogram('conv_transpose_layer/{}/kernel'.format(index), weight)
        tf.summary.histogram('conv_transpose_layer/{}/bias'.format(index), bias)

    return weight, bias, cell_out


def fc_layer(input_x, out_size, rand_seed, activation_function=None, index=0):
    """
    :param input_x: The input of the FC layer. It should be a flatten vector.
    :param in_size: The length of input vector.
    :param out_size: The length of output vector.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
    :param activation_function: The activation function for the output. Default set to None.
    :param index: The index of the layer. It is used for naming only.

    """
    shape = input_x.get_shape().as_list()

    with tf.variable_scope('fc_layer_%d' % index):
        with tf.name_scope('fc_kernel'):
            w_shape = [shape[1], out_size]
            weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        with tf.variable_scope('fc_kernel'):
            b_shape = [out_size]
            bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=rand_seed))

        cell_out = tf.add(tf.matmul(input_x, weight), bias)
        if activation_function is not None:
            cell_out = activation_function(cell_out)

        tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
        tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

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



class DCGAN(object):
    def __init__(self, session, batch_size=64, gf_dim=64, is_training=True):
    """

    """
    self.sess = session
    self.batch_size = batch_size
    self.gf_dim = gf_dim
    self.is_training = is_training

    def generator(self, z):
        """
        """
        with tf.variable_scope("Generator"):
            # Proyect and reshape
            # Project - Fully Connected layer
            self.new_z, self.h0_w, self.h0_b = fc_layer(z, out_size = self.gf_dim*8*4*4, self.seed)
            # Reshape
            conv_layer_0 = tf.reshape(self.new_z, [-1, 4, 4, self.gf_dim * 8])
            # batch normalization
            conv_layer_0 = norm_layer(conv_layer_0, self.is_training)
            # relu
            conv_layer_0 = tf.nn.relu(conv_layer_0)

            # convolution 1
            conv_layer_1 = conv_transpose_layer(input_x=conv_layer_0,
                                                output_shape=[self.batch_size, 8, 8, self.gf_dim * 4],
                                                rand_seed=self.seed,
                                                index=1)
            # batch normalization
            conv_layer_1 = norm_layer(conv_layer_1, self.is_training)
            # relu
            conv_layer_1 = tf.nn.relu(conv_layer_1)

            # convolution 2
            conv_layer_2 = conv_transpose_layer(input_x=conv_layer_1,
                                                output_shape=[self.batch_size, 16, 16, self.gf_dim * 2],
                                                rand_seed=self.seed,
                                                index=2)
            # batch normalization
            conv_layer_2 = norm_layer(conv_layer_2, self.is_training)
            # relu
            conv_layer_2 = tf.nn.relu(conv_layer_2)

            # convolution 3
            conv_layer_3 = conv_transpose_layer(input_x=conv_layer_2,
                                                output_shape=[self.batch_size, 32, 32, self.gf_dim * 1],
                                                rand_seed=self.seed,
                                                index=3)
            # batch normalization
            conv_layer_3 = norm_layer(conv_layer_3, self.is_training)
            # relu
            conv_layer_3 = tf.nn.relu(conv_layer_3)

            # convolution 4
            conv_layer_4 = conv_transpose_layer(input_x=conv_layer_3,
                                                output_shape=[self.batch_size, 64, 64, 3],
                                                rand_seed=self.seed,
                                                index=4)
            # tanh
            return tf.nn.tanh(conv_layer_4)

    def discriminator(self, image, reuse=False):
        """
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # convolution 0
        conv_layer_0 = conv_layer(image,
                                  out_channel=self.df_dim,
                                  rand_seed=self.seed,
                                  index=0)
        # leaky relu
        conv_layer_0 = tf.nn.leaky_relu(conv_layer_0, alpha=0.2)

        # convolution 1
        conv_layer_1 = conv_layer(conv_layer_1,
                                  out_channel=self.df_dim * 2,
                                  rand_seed=self.seed,
                                  index=1)
        # leaky relu
        conv_layer_1 = tf.nn.leaky_relu(conv_layer_1, alpha=0.2)

        # convolution 2
        conv_layer_2 = conv_layer(conv_layer_2,
                                  out_channel=self.df_dim * 4,
                                  rand_seed=self.seed,
                                  index=2)
        # leaky relu
        conv_layer_2 = tf.nn.leaky_relu(conv_layer_2, alpha=0.2)

        # convolution 3
        conv_layer_3 = conv_layer(conv_layer_3,
                                  out_channel=self.df_dim * 8,
                                  rand_seed=self.seed,
                                  index=3)
        # leaky relu
        conv_layer_3 = tf.nn.leaky_relu(conv_layer_3, alpha=0.2)

        # reshape
        norm_shape = conv_layer_3.get_shape()
        img_vector_length = norm_shape[1].value * norm_shape[2].value * norm_shape[3].value
        flatten = tf.reshape(conv_layer_3, shape=[-1, img_vector_length])
        # linear
        fc_layer_4, _, _ = fc_layer(flatten, out_size = 1, self.seed)

        return tf.nn.sigmoid(fc_layer_4), fc_layer_4
