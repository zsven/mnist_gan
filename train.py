from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import model
from tensorflow.examples.tutorials.mnist import input_data
from config import *
import numpy as np
from utils import *


def train(mnist):
    global_step = tf.Variable(0, trainable=False)
    x = tf.placeholder(tf.float32, [None, X_W, X_H, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, Y_SHAPE], name='output_y')
    z = tf.placeholder(tf.float32, [None, Z_SHAPE], name='z')

    with tf.variable_scope('for_reuse_scope'):
        G = model.generator(z, y_)
        D, D_logit = model.discriminator(x, y_)
        samples = model.sampler(z, y_)
        D_, D_logit_ = model.discriminator(G, y_, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_, labels=tf.ones_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_, labels=tf.ones_like(D_)))

    vars = tf.trainable_variables()
    d_vars = [var for var in vars if 'd_' in var.name]
    g_vars = [var for var in vars if 'g_' in var.name]
    d_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss=d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss=g_loss, var_list=g_vars, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(MAX_STEP):
            images, label = mnist.train.next_batch(BATCH_SIZE)
            images = tf.reshape(images, [-1, X_W, X_H, 1])
            label_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_SHAPE))
            _ = sess.run(d_optim, feed_dict={x: images, y_: label, z: label_z})
            _ = sess.run(g_optim, feed_dict={z: label_z, y_: label})
            errD_fake = d_loss_fake.eval({z:label_z, y_:label})
            errD_real = d_loss_real.eval({x: images, y_:label})
            errG = g_loss.eval({z:label_z, y_:label})
            if step % 20 == 0:
                print('step %d d_loss: %f, g_loss: %f' % (step, errD_fake+errD_real, errG))
            if step % 100 == 1 or step + 1 == MAX_STEP:
                sample = sess.run(samples, feed_dict={ z: label_z, y_: label})
                sample_path = IMG_PATH + 'test-%d.png' % step
                save_images(sample, [8, 8], sample_path)

if __name__ == '__main__':
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train(mnist)