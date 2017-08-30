import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import placeholder, constant, conv2d, upsample
from tensorflow.contrib.framework import add_arg_scope, arg_scope
from extra_layers import iconv2d, iresidual2d, iupconv2d

def fast_style(x, scope='fast_style', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([iconv2d, iresidual2d, iupconv2d], activation=tf.nn.relu):
            x = iconv2d(x, 32, 9, 1)
            x = iconv2d(x, 64, 3, 2)
            x = iconv2d(x, 128, 3, 2)
            x = iresidual2d(x, 128, 3)
            x = iresidual2d(x, 128, 3)
            x = iresidual2d(x, 128, 3)
            x = iresidual2d(x, 128, 3)
            x = iresidual2d(x, 128, 3)
            x = iupconv2d(x, 64, 3, 2)
            x = iupconv2d(x, 32, 3, 2)
            x = iconv2d(x, 3, 9, 1, activation=None)
            with tf.name_scope('rescale'):
                output = (tf.nn.tanh(x) + 1) * 255./2

    return output
