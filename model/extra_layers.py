import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import placeholder, constant, conv2d
from tensorflow.contrib.framework import add_arg_scope, arg_scope

@add_arg_scope
def instance_norm(x, eps=1e-3, scope=None, reuse=None):
    # Expects 4D Tensor
    C = x._shape_as_list()[3]

    with tf.variable_scope(scope, 'instance_norm', reuse=reuse):
        # Get learnable parameters
        beta = tf.get_variable('beta', C, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', C, initializer=tf.ones_initializer)
        # Normalize each sample separately across (H, W) and rescale
        m, v = tf.nn.moments(x, [1, 2], keep_dims=True)
        norm = (x - m) * tf.rsqrt(v + eps)
        output = gamma * norm + beta

    return output

@add_arg_scope
def iconv2d(x, num_outputs, kernel_size, strides, scope=None, reuse=None, activation=None):
    with tf.variable_scope(scope, 'iconv2d', reuse=None):
        padding = [0, kernel_size / 2, kernel_size / 2, 0]
        padding = [[v, v] for v in padding]
        x = tf.pad(x, padding, mode='REFLECT')
        x = conv2d(x, num_outputs, kernel_size, strides, padding='VALID')
        output = instance_norm(x)

        if activation:
            output = activation(output)

    return output


@add_arg_scope
def iresidual2d(x, num_outputs, kernel_size, scope=None, reuse=None, activation=None):
    with tf.variable_scope(scope, 'iresidual2d', reuse=None):
        res = iconv2d(x, num_outputs, kernel_size, 1, activation=activation)
        res = iconv2d(res, num_outputs, kernel_size, 1, activation=None)
        output = x + res

    return output

def upsample(x,
             strides,
             scope=None):
    # Convert int to list
    strides = [strides] * 2 if isinstance(strides, int) else strides
    H, W = x._shape_as_list()[1:3]
    with tf.variable_scope(scope, 'upsample'):
        if H is None or W is None:
            _, H, W, _ = tf.unstack(tf.shape(x))
        H, W = strides[0] * H, strides[1] * W
        output = tf.image.resize_nearest_neighbor(x, [H, W])
    return output

@add_arg_scope
def iupconv2d(x, num_outputs, kernel_size, strides, scope=None, reuse=None, activation=None):
    with tf.variable_scope(scope, 'ideconv2d', reuse=None):
        x = upsample(x, strides)
        output = iconv2d(x, num_outputs, kernel_size, 1, activation=activation)

    return output
