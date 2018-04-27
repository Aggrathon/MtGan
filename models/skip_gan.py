"""
    All the state of the art models seems to ignore the input.
    This is an attempt at avoiding that with lot of skip connections
"""

import tensorflow as tf
from models.model import get_art_only_cropped, Generator

BATCH_SIZE = 32
CODE_SIZE = 200
WIDTH = 5*8
HEIGHT = 4*8

def _generator_cell(prev_layer, size, data, height, width, training, name, norm=True):
    with tf.variable_scope(name):
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
        if data is not None:
            skip = tf.layers.dense(data, (width+2)*(height+2)*4, activation=tf.nn.relu)
            skip = tf.reshape(skip, (BATCH_SIZE, height+2, width+2, 4))
            prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same') + tf.layers.conv2d(skip, size, 3, 1, 'valid')
            prev_layer = tf.nn.relu(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*2, 1, 1)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
        if norm:
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
    return prev_layer


def _generator(data, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        prev_layer = tf.reshape(tf.layers.dense(data, 4*5*128, activation=tf.nn.relu), (BATCH_SIZE, 4, 5, 128), 'layer0')
        prev_layer = _generator_cell(prev_layer, 1024, data, 4, 5, training, 'layer1')
        prev_layer = _generator_cell(prev_layer, 768, data, 8, 10, training, 'layer2')
        prev_layer = _generator_cell(prev_layer, 512, data, 16, 20, training, 'layer3')
        #prev_layer = _generator_cell(prev_layer, 512, data, 32, 40, training, 'layer4')
        with tf.variable_scope('layer6'):
            prev_layer = tf.layers.conv2d(prev_layer, 256, 5, 1, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 1, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(prev_layer, 3, 1, 1, 'same', activation=tf.nn.tanh)
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

def _discriminator_cell(prev_layer, skip, training, size, name, norm=True):
    with tf.variable_scope(name):
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
        if skip is not None:
            prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same') + tf.layers.conv2d(skip, size, 3, 2, 'same')
            prev_layer = tf.nn.relu(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*2, 1, 1)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
        if norm:
            with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
                prev_layer = tf.contrib.layers.layer_norm(prev_layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 2, 'same', activation=tf.nn.relu)
        return prev_layer

def _discriminator(data, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        layer0 = _discriminator_cell(data, None, training, 256, 'layer0')
        layer1 = _discriminator_cell(layer0, data, training, 512, 'layer1')
        layer2 = _discriminator_cell(layer1, layer0, training, 512, 'layer2')
        layer3 = layer2#  _discriminator_cell(layer2, layer1, training, 1024, 'layer3')
        with tf.variable_scope('layer5'):
            wgan = tf.layers.conv2d(layer3, 512, 1, 1, 'same', activation=tf.nn.relu)
            wgan = tf.layers.conv2d(wgan, 256, 3, 1, 'valid', activation=tf.nn.relu)
            wgan = tf.layers.flatten(wgan)
            wgan = tf.layers.dense(wgan, 1, name='logits_wgan')
            dcgan = tf.layers.conv2d(layer3, 512, 1, 1, 'same', activation=tf.nn.relu)
            dcgan = tf.layers.conv2d(dcgan, 256, 3, 1, 'valid', activation=tf.nn.relu)
            dcgan = tf.layers.flatten(dcgan)
            dcgan = tf.layers.dense(dcgan, 2, name='logits_dcgan')
        if reuse:
            return wgan, dcgan
        else:
            return wgan, dcgan, scope.trainable_variables()

class SkipGanGenerator(Generator):
    """
        Skip Gan Neural Network
    """

    def __init__(self, training=True):
        super().__init__('skip-gan')
        with self.scope:
            self.seed = tf.random_uniform((BATCH_SIZE, CODE_SIZE), -1.0, 1.0)
            self.real_image = tf.image.resize_bilinear(get_art_only_cropped(batch_size=BATCH_SIZE), (HEIGHT, WIDTH))
            self.fake_image, gen_vars = _generator(self.seed, False, training)
            self.real_critic, self.real_disc, disc_vars = _discriminator(self.real_image, False, training)
            self.fake_critic, self.fake_disc = _discriminator(self.fake_image, True, training)
            #gp
            x_hat = self.real_image + tf.random_uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0) * (self.fake_image - self.real_image)
            y_hat, _ = _discriminator(x_hat, True, training)
            gradients = tf.gradients(y_hat, x_hat)
            slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slope-1.0), name='gradient_penalty')
            #dc
            dc_loss_rr = tf.losses.softmax_cross_entropy([[0.9, 0.1]]*BATCH_SIZE, self.real_disc)
            dc_loss_rf = tf.losses.softmax_cross_entropy([[0.1, 0.9]]*BATCH_SIZE, self.fake_disc)
            dc_loss_ff = tf.losses.softmax_cross_entropy([[0.9, 0.1]]*BATCH_SIZE, self.fake_disc)
            #losses
            rmf = tf.reduce_mean(self.fake_critic)
            rmr = tf.reduce_mean(self.real_critic)
            diff = rmr - rmf
            #self.loss_g = tf.losses.compute_weighted_loss([rmf, dc_loss_ff], [-1.0, 0.5])
            #self.loss_d = tf.losses.compute_weighted_loss(
            #    [rmf, rmr, self.gradient_penalty, dc_loss_rf, dc_loss_rr],
            #   [1.0, -1.0, 10.0, 0.5, 0.5])
            self.loss_g = tf.losses.compute_weighted_loss([dc_loss_ff], [1.0])
            self.loss_d = tf.losses.compute_weighted_loss([dc_loss_rf, dc_loss_rr], [1.0, 1.0])
            #meta
            self.distance = tf.get_variable('distance', [], tf.float32, trainable=False, initializer=tf.initializers.ones)
            distance = tf.assign(self.distance, 0.9*self.distance + 0.1*diff)
            #training
            learning_rate = tf.multiply(tf.to_float(self.global_step), 3e-4)
            learning_rate = tf.sqrt(learning_rate)
            learning_rate = tf.pow(10.0, -(4.5+learning_rate), 'learning_rate')
            trainer = tf.train.AdamOptimizer(learning_rate, 0.05, 0.9, name='Adam')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.trainer_d = trainer.minimize(
                    self.loss_d,
                    var_list=disc_vars,
                    name='train_d'
                )
                trainer_g = trainer.minimize(
                    self.loss_g,
                    global_step=self.global_step,
                    var_list=gen_vars,
                    name='train_g'
                )
            self.trainer_g = tf.group(self.trainer_d, trainer_g, distance)
            #summary
            tf.summary.scalar("Distance", distance)
            tf.summary.scalar("GradientPenalty", self.gradient_penalty)
            tf.summary.scalar("LossD", self.loss_d)
            tf.summary.scalar("LossG", self.loss_g)
            tf.summary.image("Generated", self.fake_image, 4)

    def train_step(self, summary=False, session=None):
        """
            Do a training step
        """
        if session is None:
            session = self.session
        for _ in range(5):
            session.run(self.trainer_d)
        if summary:
            _, smry, step, res = session.run([self.trainer_g, self.summary, self.global_step, self.distance])
            self.add_summary(smry, step)
        else:
            _, step, res = session.run([self.trainer_g, self.global_step, self.distance])
        return step, res
