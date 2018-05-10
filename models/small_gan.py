"""
    Other models doesn't seem to converge this is an attempt at a basic
    network that avoids that (less romm for running in circles).
"""

import tensorflow as tf
from models.model import get_art_only_cropped, Generator

BATCH_SIZE = 128
CODE_SIZE = 200
WIDTH = 5*16
HEIGHT = 4*16
DTYPE = tf.float32

def _generator(data, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        size = 64
        prev_layer = tf.reshape(tf.layers.dense(data, 4*5*size*4, activation=tf.nn.relu), (BATCH_SIZE, 4, 5, size*4), 'layer0')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size*8, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size*4, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size*2, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 32, 3, 1, 'same')
        prev_layer = tf.layers.conv2d(prev_layer, 3, 1, 1, 'same', activation=tf.nn.tanh)
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

def _discriminator(data, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        size = 64
        prev_layer = tf.layers.conv2d(data, size, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*2, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*4, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*8, 3, 2, 'same', activation=tf.nn.relu)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size*4, 3, 1, 'same', activation=tf.nn.relu)
        prev_layer = tf.layers.flatten(prev_layer)
        prev_layer = tf.layers.dense(prev_layer, 1, name='logits_wgan')
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

class SmallGanGenerator(Generator):
    """
        Small Gan Neural Network
    """

    def __init__(self, training=True):
        super().__init__('small-gan')
        with self.scope:
            self.seed = tf.random_uniform((BATCH_SIZE, CODE_SIZE), -1.0, 1.0, DTYPE)
            self.real_image = tf.cast(tf.image.resize_bilinear( \
                get_art_only_cropped(batch_size=BATCH_SIZE), (HEIGHT, WIDTH)), DTYPE)
            self.fake_image, gen_vars = _generator(self.seed, False, training)
            self.real_critic, disc_vars = _discriminator(self.real_image, False, training)
            self.fake_critic = _discriminator(self.fake_image, True, training)
            #gp
            x_hat = self.real_image + tf.random_uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0, DTYPE) * (self.fake_image - self.real_image)
            y_hat = _discriminator(x_hat, True, training)
            gradients = tf.gradients(y_hat, x_hat)
            slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slope-10.0), name='gradient_penalty')
            #losses
            rmf = tf.reduce_mean(self.fake_critic)
            rmr = tf.reduce_mean(self.real_critic)
            diff = rmr - rmf
            self.loss_g = tf.losses.compute_weighted_loss([rmf], [-1.0])
            self.loss_d = tf.losses.compute_weighted_loss([rmf, rmr, self.gradient_penalty], [1.0, -1.0, 10.0])
            #meta
            self.distance = tf.get_variable('distance', [], DTYPE, trainable=False, initializer=tf.initializers.ones)
            distance = tf.assign(self.distance, 0.9*self.distance + 0.1*diff)
            #training
            learning_rate = tf.multiply(tf.to_float(self.global_step), 3e-4)
            learning_rate = tf.sqrt(learning_rate)
            learning_rate = tf.pow(10.0, -(3.8+learning_rate), 'learning_rate')
            trainer_g = tf.train.AdamOptimizer(learning_rate*0.5, 0, 0.9, name='AdamG')
            trainer_d = tf.train.AdamOptimizer(learning_rate, 0, 0.9, name='AdamD')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.trainer_d = trainer_d.minimize(
                    self.loss_d,
                    var_list=disc_vars,
                    name='train_d'
                )
                trainer_g = trainer_g.minimize(
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
        #print(session.run([self.gradient_penalty, self.distance, self.real_critic, self.fake_critic]))
        for _ in range(5):
            session.run(self.trainer_d)
        if summary:
            _, smry, step, res = session.run([self.trainer_g, self.summary, self.global_step, self.distance])
            self.add_summary(smry, step)
        else:
            _, step, res = session.run([self.trainer_g, self.global_step, self.distance])
        return step, res