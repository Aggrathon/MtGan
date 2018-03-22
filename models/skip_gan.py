"""
    All the state of the art models seems to ignore the input.
    This is an attempt at avoiding that with lot of skip connections
"""

import tensorflow as tf
from models.model import IMAGE_HEIGHT, IMAGE_WIDTH, get_art_only_data, Generator

BATCH_SIZE = 32
CODE_SIZE = 400
LEARNING_RATE = 2e-5

def _generator(data, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        prev_layer = tf.reshape(tf.layers.dense(data, 8*11*8, activation=tf.nn.relu), (BATCH_SIZE, 8, 11, 8), 'layer0')
        with tf.variable_scope('layer1'):
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 1, 'same', activation=tf.nn.relu)
            skip = tf.reshape(tf.layers.dense(data, 10*13, activation=tf.nn.relu), (BATCH_SIZE, 10, 13, 1))
            prev_layer = prev_layer + tf.layers.conv2d(skip, 128, 3, 1, 'valid')
            prev_layer = tf.layers.conv2d_transpose(prev_layer, 256, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        with tf.variable_scope('layer2'):
            prev_layer = tf.layers.conv2d(prev_layer, 256, 3, 1, 'same', activation=tf.nn.relu)
            skip = tf.reshape(tf.layers.dense(data, 18*24, activation=tf.nn.relu), (BATCH_SIZE, 18, 24, 1))
            prev_layer = prev_layer + tf.layers.conv2d(skip, 256, 3, 1, 'valid')
            prev_layer = tf.layers.conv2d_transpose(prev_layer, 256, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        with tf.variable_scope('layer3'):
            prev_layer = tf.layers.conv2d(prev_layer, 256, 3, 1, 'same', activation=tf.nn.relu)
            skip = tf.reshape(tf.layers.dense(data, 34*46, activation=tf.nn.relu), (BATCH_SIZE, 34, 46, 1))
            prev_layer = prev_layer + tf.layers.conv2d(skip, 256, 3, 1, 'valid')
            prev_layer = tf.layers.conv2d_transpose(prev_layer, 256, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        with tf.variable_scope('layer4'):
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 1, 'same', activation=tf.nn.relu)
            skip = tf.reshape(tf.layers.dense(data, 66*90, activation=tf.nn.relu), (BATCH_SIZE, 66, 90, 1))
            prev_layer = prev_layer + tf.layers.conv2d(skip, 128, 3, 1, 'valid')
            prev_layer = tf.layers.conv2d_transpose(prev_layer, 128, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        with tf.variable_scope('layer5'):
            prev_layer = tf.layers.conv2d(prev_layer, 64, 3, 1, 'same', activation=tf.nn.relu)
            skip = tf.reshape(tf.layers.dense(data, 130*178, activation=tf.nn.relu), (BATCH_SIZE, 130, 178, 1))
            prev_layer = prev_layer + tf.layers.conv2d(skip, 64, 3, 1, 'valid')
            prev_layer = tf.layers.conv2d(prev_layer, 3, 5, 1, 'same', activation=tf.nn.tanh)
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

def _discriminator(data, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        with tf.variable_scope('layer0'):
            prev_layer = tf.layers.conv2d(data, 32, 3, 1, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(prev_layer, 32, 3, 2, 'same', activation=tf.nn.relu)
            with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
                prev_layer = tf.contrib.layers.layer_norm(prev_layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
            skip1 = prev_layer
        with tf.variable_scope('layer1'):
            prev_layer = tf.layers.conv2d(prev_layer, 64, 3, 1, 'same', activation=tf.nn.relu)
            prev_layer = prev_layer + tf.layers.conv2d(data, 64, 3, 2, 'same')
            prev_layer = tf.layers.conv2d(prev_layer, 64, 3, 2, 'same')
            with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
                prev_layer = tf.contrib.layers.layer_norm(prev_layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
            skip2 = prev_layer
        with tf.variable_scope('layer2'):
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 1, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(skip1, 128, 3, 2, 'same')
            with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
                prev_layer = tf.contrib.layers.layer_norm(prev_layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
        with tf.variable_scope('layer3'):
            prev_layer = tf.layers.conv2d(prev_layer, 256, 3, 1, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(prev_layer, 256, 3, 2, 'same', activation=tf.nn.relu)
            prev_layer = tf.layers.conv2d(skip2, 128, 3, 2, 'same')
            with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
                prev_layer = tf.contrib.layers.layer_norm(prev_layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
            prev_layer = tf.layers.conv2d(prev_layer, 128, 3, 1, 'same', activation=tf.nn.relu)
        with tf.variable_scope('layer4'):
            prev_layer = tf.layers.flatten(prev_layer)
            prev_layer = tf.layers.dropout(prev_layer, 0.3, training=training)
            prev_layer = tf.layers.dense(prev_layer, 256, activation=tf.nn.relu)
            prev_layer = tf.layers.dropout(prev_layer, 0.3, training=training)
            prev_layer = tf.layers.dense(prev_layer, 1, name='logits')
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

class SkipGanGenerator(Generator):
    """
        Skip Gan Neural Network
    """

    def __init__(self, training=True):
        super().__init__('skip-gan')
        with self.scope:
            self.seed = tf.random_uniform((BATCH_SIZE, CODE_SIZE), -1.0, 1.0)
            self.real_image = get_art_only_data(batch_size=BATCH_SIZE)
            self.fake_image, gen_vars = _generator(self.seed, False, training)
            self.real_disc, disc_vars = _discriminator(self.real_image, False, training)
            self.fake_disc = _discriminator(self.fake_image, True, training)
            #gp
            x_hat = self.real_image + tf.random_uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0) * (self.fake_image - self.real_image)
            y_hat = _discriminator(x_hat, True, training)
            gradients = tf.gradients(y_hat, x_hat)
            slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slope-1.0), name='gradient_penalty')
            #losses
            rmf = tf.reduce_mean(self.fake_disc)
            rmr = tf.reduce_mean(self.real_disc)
            diff = tf.maximum(0.0, rmr - rmf)
            self.loss_g = tf.negative(rmf, name='loss_g')
            self.loss_d = tf.losses.compute_weighted_loss( [rmf, rmr, self.gradient_penalty], [1.0, -1.0, 10.0])
            #meta
            self.distance = tf.get_variable('distance', [], tf.float32, trainable=False, initializer=tf.initializers.ones)
            self.measure = tf.get_variable('measure', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
            measure = tf.assign(self.measure, 0.95*self.measure-0.05*self.loss_d)
            distance = tf.assign(self.distance, 0.95*self.distance + 0.05*diff)
            #training
            adam = tf.train.AdamOptimizer(LEARNING_RATE, 0.0, 0.9, name='Adam')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.trainer_d = adam.minimize(
                    self.loss_d,
                    var_list=disc_vars,
                    name='train_d'
                )
                trainer_g = adam.minimize(
                    self.loss_g,
                    global_step=self.global_step,
                    var_list=gen_vars,
                    name='train_g'
                )
            self.trainer_g = tf.group(self.trainer_d, trainer_g, measure, distance)
            #summary
            tf.summary.scalar("Measure", measure)
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
        for _ in range(10):
            session.run(self.trainer_d)
        if summary:
            _, smry, step, res = session.run([self.trainer_g, self.summary, self.global_step, self.measure])
            self.add_summary(smry, step)
        else:
            _, step, res = session.run([self.trainer_g, self.global_step, self.measure])
        return step, res
