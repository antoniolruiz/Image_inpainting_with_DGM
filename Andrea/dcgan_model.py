#!/usr/bin/env python
import tensorflow as tf


def conv_layer(input_x, in_channel, out_channel, stride, kernel_shape, rand_seed, index=0):
    """
    :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
    :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
    :param index: The index of the layer. It is used for naming only.
    """
    assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

    with tf.variable_scope('conv_layer_%d' % index):
        with tf.name_scope('conv_kernel'):
            w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
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
    return weight, bias, cell_out



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


def linear_layer(input_x, output_size, stddev=0.02, bias_start=0.0):
    shape = input_x.get_shape().as_list()

    with tf.variable_scope("Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_x, matrix) + bias, matrix, bias


class DCGAN(object):
    def __init__(self, session, batch_size=64, gf_dim=64, is_training=True):
    """

    """
    self.sess = session
    self.batch_size = batch_size
    self.gf_dim = gf_dim
    self.is_training = is_training

    def generator(self, input_x):
        """
        """
        with tf.variable_scope("Generator"):
            # Proyect and reshape
            # Project - Lineal layer
            self.z, self.h0_w, self.h0_b = linear_layer(input_x, self.gf_dim*8*4*4)
            # Reshape
            conv_layer_0 = tf.reshape(self.z, [-1, 4, 4, self.gf_dim * 8])
            # batch normalization
            conv_layer_0 = norm_layer(conv_layer_0, self.is_training)
            # relu
            conv_layer_0 = tf.nn.relu(conv_layer_0)

            # convolution 1
            _, _, conv_layer_1 = conv_transpose_layer(input_x=conv_layer_0,
                                                output_shape=[self.batch_size, 8, 8, self.gf_dim * 4],
                                                rand_seed=self.seed,
                                                index=1)
            # batch normalization
            conv_layer_1 = norm_layer(conv_layer_1, self.is_training)
            # relu
            conv_layer_1 = tf.nn.relu(conv_layer_1)

            # convolution 2
            _, _, conv_layer_2 = conv_transpose_layer(input_x=conv_layer_1,
                                                output_shape=[self.batch_size, 16, 16, self.gf_dim * 2],
                                                rand_seed=self.seed,
                                                index=2)
            # batch normalization
            conv_layer_2 = norm_layer(conv_layer_2, self.is_training)
            # relu
            conv_layer_2 = tf.nn.relu(conv_layer_2)

            # convolution 3
            _, _, conv_layer_3 = conv_transpose_layer(input_x=conv_layer_2,
                                                output_shape=[self.batch_size, 32, 32, self.gf_dim * 1],
                                                rand_seed=self.seed,
                                                index=3)
            # batch normalization
            conv_layer_3 = norm_layer(conv_layer_3, self.is_training)
            # relu
            conv_layer_3 = tf.nn.relu(conv_layer_3)

            # convolution 4
            _, _, conv_layer_4 = conv_transpose_layer(input_x=conv_layer_3,
                                                output_shape=[self.batch_size, 64, 64, 3],
                                                rand_seed=self.seed,
                                                index=4)
            # tanh
            return tf.nn.tanh(conv_layer_4)

