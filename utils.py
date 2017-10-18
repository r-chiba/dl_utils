from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf
import numpy as np

def pad_out_size_same(in_size, stride):
    return int(math.ceil(float(in_size) / float(stride)))

def pad_out_size_valid(in_size, filter_size, stride):
    return int(math.ceil(float(in_size - filter_size + 1) / float(stride)))

def conv2d(input_, output_dim, kh=5, kw=5, sth=1, stw=1, sd=0.02, padding='SAME',
            bias=True, name='conv2d', with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=sd))
        conv = tf.nn.conv2d(input_, w, strides=[1, sth, stw, 1], padding=padding)
        if bias:
            bias = tf.get_variable('bias', [output_dim],
                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
            if with_w:
                return conv, w, bias
            else:
                return conv
        else:
            if with_w:
                return conv, w
            else:
                return conv

def deconv2d(input_, output_shape, kh=5, kw=5, sth=1, stw=1, sd=0.02, padding='SAME',
            bias=True, name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=sd))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, sth, stw, 1], padding=padding)
        if bias:
            bias = tf.get_variable('bias', [output_shape[-1]],
                    initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, bias)
            if with_w:
                return deconv, w, bias
            else:
                return deconv
        else:
            if with_w:
                return deconv, w
            else:
                return deconv

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, vs_name='Linear', sd=0.02, bias=True, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(vs_name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=sd))
        if bias:
            b = tf.get_variable('b', [output_size],
                    initializer=tf.constant_initializer(bias_start))
            if with_w == True:
                return tf.matmul(input_, w) + b, w, b
            else:
                return tf.matmul(input_, w) + b
        else:
            if with_w == True:
                return tf.matmul(input_, w), w
            else:
                return tf.matmul(input_, w)

def blur(batch):
    filter_oc = [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
    filter_oc = np.array(filter_oc, dtype=np.float32)
    filter_oc /= 256.
    filter_oc = filter_oc.reshape((5, 5, 1, 1))
    
    n_channel = batch.get_shape().as_list()[-1]
    if n_channel > 1:
        filter_ = np.repeat(filter_oc, n_channel, axis=2)
    else:
        filter_ = filter_oc

    blured = tf.nn.depthwise_conv2d(batch, filter_, 
        strides=[1, 1, 1, 1], padding='SAME')
    return blured

def get_gaussian_pyramid(batch, level):
    ret = [batch]
    batch_cur = batch
    for i in xrange(level-1):
        blured = blur(batch_cur)
        batch_cur = tf.nn.max_pool(batch_cur, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        ret.append(batch_cur)
    return ret

def get_laplacian_pyramid(batch, level):
    ret = []
    gp = get_gaussian_pyramid(batch, level+1)
    batch_cur = batch
    for i in xrange(level):
        img = gp[i]
        size = img.get_shape()[1:3]
        img_us = tf.image.resize_images(gp[i+1], size)
        ret.append(img - img_us)
    return ret

def laplacian_pyramid_loss(batch1, batch2, level):
    lp1 = get_laplacian_pyramid(batch1, level)
    lp2 = get_laplacian_pyramid(batch2, level)
    ret = 0.
    for i in xrange(level):
        ret += 2**(-2*(i+1)) * tf.reduce_mean(tf.abs(lp1[i] - lp2[i]))
    return ret

