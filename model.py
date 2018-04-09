from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np

trunc_norm = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)
constant = lambda value: tf.constant_initializer(value=value)


def _batch_norm_layer(value, is_train=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
     return batch_norm(value, is_training=is_train, scope=scope)


def _fc_layers(value, output_shape, name='fc', stddev=0.1):
    shape = value.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [shape[1], output_shape], dtype=tf.float32, initializer=trunc_norm(stddev))
        biases = tf.get_variable('biases', [output_shape], dtype=tf.float32, initializer=constant(0.0))
        fc = tf.matmul(value, weight) + biases
        return fc


def conv_cond_concat(value, cond, name='concat'):
    first_shape = tf.stack(tf.shape(value)[0])
    value_shape = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        return tf.concat([value, cond * tf.ones(shape=[first_shape, value_shape[1], value_shape[2], cond_shapes[3]])], 3, name=name)


def _deconv2d(value, output_shape, kernel=5, name='deconv2d'):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weight', [kernel, kernel, output_shape[-1], value.get_shape()[-1]],
                                  dtype=tf.float32, initializer=trunc_norm(stddev=0.01))
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=[1, 2, 2, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], dtype=tf.float32, initializer=constant(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        return deconv


def _conv2d(value, output_shape, kernel=3, name='conv2d'):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weight', [kernel, kernel, value.get_shape().as_list()[-1], output_shape], dtype=tf.float32,
                            initializer=trunc_norm(stddev=0.01))
        b = tf.get_variable('biases', [output_shape], dtype=tf.float32, initializer=constant(0.0))
        conv = tf.nn.conv2d(value, w, strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)
        return conv


def generator(z, y):
    batch_size = tf.stack(tf.shape(z))[0]
    yb = tf.reshape(y, [-1, 1, 1, 10], name='g_y')
    z = tf.concat([z, y], 1)
    conv1 = tf.nn.relu(_batch_norm_layer(_fc_layers(z, 1024, name='g_fc1'), name='g_bn1'))
    conv1 = tf.concat([conv1, y], 1, name='g_concat1')

    conv2 = tf.nn.relu(_batch_norm_layer(_fc_layers(conv1, 128 * 7 * 7, name='g_fc2'), name='g_bn2'))
    conv2 = tf.reshape(conv2, [-1, 7, 7, 128], name='g_reshape')
    conv2 = conv_cond_concat(conv2, yb, name='g_conv_cond_concat1')

    conv3 = tf.nn.relu(_batch_norm_layer(_deconv2d(conv2, [batch_size, 14, 14, 128], name='g_deconv2d1'), name='g_bn3'))
    conv3 = conv_cond_concat(conv3, yb, name='g_conv_cond_concat2')
    net = tf.nn.sigmoid(_deconv2d(conv3, [batch_size, 28, 28, 1], name='g_deconv2'), name='g_image')
    return net


def discriminator(image, y, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    batch_size = tf.stack(tf.shape(y))[0]
    yb = tf.reshape(y, [-1, 1, 1, 10], name='d_y')
    x = conv_cond_concat(image, yb, name='d_conv_cond_concat1')
    conv1 = _conv2d(x, 11, name='d_conv1')
    lrelu1 = tf.maximum(conv1, 0.2 * conv1, name='d_lrelu1')
    conv1 = conv_cond_concat(lrelu1, yb, name='d_conv_cond_concat1')

    bn2 = _batch_norm_layer(_conv2d(conv1, 74, name='d_conv2d2'), name='d_bn2')
    lrelu2 = tf.maximum(bn2, 0.2 * bn2, name='d_lrelu2')
    shape = lrelu2.get_shape().as_list()
    conv2 = tf.reshape(lrelu2, [-1, shape[1] * shape[2] * shape[3]], name='d_reshape2')
    concat2 = tf.concat([conv2, y], 1, name='d_concat2')

    bn3 = _batch_norm_layer(_fc_layers(concat2, 1024, name='d_fc3'))
    l3 = tf.maximum(bn3, 0.2 * bn3, name='d_lrelu3')
    concat3 = tf.concat([l3, y], 1, name='d_concat3')

    fc4 = _fc_layers(concat3, 1, name='d_fc4')
    return tf.nn.sigmoid(fc4, name='d_sigmoid4'), fc4


def sampler(z, y):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y)



