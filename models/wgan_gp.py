"""
    The WGAN-GP neural network model
	https://arxiv.org/abs/1704.00028
"""
import tensorflow as tf
from models.model import IMAGE_HEIGHT, IMAGE_WIDTH, get_art_only_data, Generator

BATCH_SIZE = 64
CODE_SIZE = 100
LAYERS = 4
LEARNING_RATE = 1e-4

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

class ArtGeneratorWGANGP(Generator):
    """
        Neural network for generating art based on the WGAN-GP architechture
    """

    @classmethod
    def _generator(cls, input_data, training=True, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
        """
            The generator neural network for the WGAN-GP architechture
        """
        with tf.variable_scope('generator') as scope:
            width = image_width // (1<<4)
            height = image_height // (1<<4)
            layer = tf.layers.dense(input_data, height*width*256, activation=tf.nn.elu, name='dense')
            layer = tf.reshape(layer, (-1, height, width, 256))
            first_layer = layer
            layer = conv2d_upscale(layer, 256, training, None, 'layer_0')
            layer = conv2d_upscale(layer, 192, training, first_layer, 'layer_1')
            layer = conv2d_upscale(layer, 128, training, None, 'layer_2')
            layer = conv2d_upscale(layer, 64, training, first_layer, 'layer_3')
            layer = tf.layers.conv2d(layer, 3, 5, 1, 'same', activation=tf.nn.tanh, name='conv2d_final')
            return layer, scope.trainable_variables()

    @classmethod
    def _discriminator(cls, images, reuse, training=True):
        """
            The discriminator neural network for the WPGAN-GP architechture
        """
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            layer0 = conv2d_downscale(images, 64, None, 'layer_0')
            layer1 = conv2d_downscale(layer0, 96, None, 'layer_1')
            layer2 = conv2d_downscale(layer1, 128, layer0, 'layer_2')
            layer3 = conv2d_downscale(layer2, 160, layer1, 'layer_3')
            #layer dense
            layer = tf.layers.flatten(layer3)
            layer = tf.layers.dense(layer, 128, activation=tf.nn.elu, name='dense')
            layer = tf.layers.dropout(layer, 0.3, training=training)
            layer = tf.layers.dense(layer, 1, name='logits')
            if reuse:
                return layer
            return layer, scope.trainable_variables()

    def __init__(self, training=True, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, input_size=CODE_SIZE):
        super().__init__('wgan-gp')
        with self.scope:
            #generator and critic
            self.seed = tf.random_uniform((batch_size, input_size), -1.0, 1.0)
            self.real_image = get_art_only_data(batch_size=batch_size)
            self.fake_image, self.generator_variables = ArtGeneratorWGANGP._generator(self.seed, training)
            self.real_disc, self.discriminator_variables = ArtGeneratorWGANGP._discriminator(self.real_image, False, training)
            self.fake_disc = ArtGeneratorWGANGP._discriminator(self.fake_image, True, training)
            #gradient penalty
            x_hat = self.real_image + tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0) * (self.fake_image - self.real_image)
            y_hat = ArtGeneratorWGANGP._discriminator(x_hat, True, False)
            gradients = tf.gradients(y_hat, x_hat)
            slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slope-1.0), name='gradient_penalty')
            #losses
            self.loss_g = -tf.reduce_mean(self.fake_disc, name='loss_g')
            self.loss_d = tf.losses.compute_weighted_loss(
                [tf.reduce_mean(self.fake_disc), tf.reduce_mean(self.real_disc), self.gradient_penalty, tf.reduce_mean(tf.square(self.real_disc))],
                [1.0, -1.0, 10.0, 0.001]
            )
            adam_d = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamD')
            adam_g = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamG')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                trainer_d = adam_d.minimize(self.loss_d, var_list=self.discriminator_variables, name='train_d')
                self.trainer_g = adam_g.minimize(self.loss_g, global_step=self.global_step,
                                                 var_list=self.generator_variables, name='train_g')
            #summary
            self.measure = tf.get_variable('measure', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
            measure = tf.assign(self.measure, 0.95*self.measure-0.05*self.loss_d)
            tf.summary.scalar("Measure", measure)
            tf.summary.scalar("GradientPenalty", self.gradient_penalty)
            tf.summary.scalar("LossD", self.loss_d)
            tf.summary.scalar("LossG", self.loss_g)
            tf.summary.image("Generated", self.fake_image, 4)
            self.trainer_d = tf.group(trainer_d, measure)

    def train_step(self, summary=False, session=None):
        """
            Do a training step
        """
        if session is None:
            session = self.session
        for _ in range(4):
            session.run(self.trainer_d)
        if summary:
            _, smry, step, res = session.run([self.trainer_g, self.summary, self.global_step, self.measure])
            self.add_summary(smry, step)
        else:
            _, step, res = session.run([self.trainer_g, self.global_step, self.measure])
        return step, res
