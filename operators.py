"""
    Script for reusable operators
"""
import tensorflow as tf

def conv2d_upscale(prev_layer, kernels, training=True, shortcut=None, name='conv2d_upscale'):
    """
        Do a upscaling convolution with an optional shortcut
    """
    with tf.variable_scope(name):
        layer = tf.layers.conv2d(prev_layer, kernels, 5, 1, 'same', name='conv2d')
        if shortcut is not None:
            shape1 = int(layer.get_shape()[1]) // int(shortcut.get_shape()[1])
            shape2 = int(layer.get_shape()[2]) // int(shortcut.get_shape()[2])
            shortcut = tf.layers.conv2d_transpose(shortcut, kernels, (shape1+1, shape2+1), \
                (shape1, shape2), 'same', name='shortcut')
            layer = layer + shortcut
        layer = tf.layers.conv2d_transpose(layer, kernels, 3, 2, 'same', activation=tf.nn.elu, name='resize')
        layer = tf.layers.batch_normalization(layer, -1, training=training)
        return layer

def conv2d_downscale(prev_layer, kernels, shortcut=None, name='conv2d_downscale'):
    """
        Do a downscaling convolution with an optional shortcut
    """
    with tf.variable_scope(name):
        layer = tf.layers.conv2d(prev_layer, kernels, 5, 1, 'same', name='conv2d')
        if shortcut is not None:
            shape1 = int(shortcut.get_shape()[1]) // int(layer.get_shape()[1])
            shape2 = int(shortcut.get_shape()[2]) // int(layer.get_shape()[2])
            shortcut = tf.layers.conv2d(shortcut, kernels, (shape1+1, shape2+1), (shape1, shape2), \
                'same', name='shortcut')
            layer = layer + shortcut
        layer = tf.layers.conv2d(layer, kernels, 3, 2, 'same', activation=tf.nn.elu, name='resize')
        with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
            layer = tf.contrib.layers.layer_norm(layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
        return layer
            