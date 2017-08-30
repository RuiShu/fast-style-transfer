# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Modified by Rui Shu

import tensorflow as tf
import numpy as np
import scipy.io
from config import args

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

def vgg_features(x, scope='vgg', reuse=None, get_style=False, get_content=False):
    style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    content_layers = {'relu4_2'}

    style = {}
    content = {}

    with tf.variable_scope(scope, reuse=reuse):
        x = preprocess(x)
        net = vgg(args.vgg_path, x)

        if get_style:
            for layer in style_layers:
                h = net[layer]
                H, W, C = h._shape_as_list()[1:]
                h_size = H * W * C
                h = tf.reshape(h, (-1, H * W, C))
                h_T = tf.transpose(h, perm=[0, 2, 1])
                # Normalized Gram matrix (aka second moment)
                style[layer] = tf.matmul(h_T, h) / h_size

        if get_content:
            for layer in content_layers:
                content[layer] = net[layer]

    return style, content

def vgg(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL
