"""
    This model builds upon the small_gan but adds residual layers.
"""

import tensorflow as tf
from models.model import get_art_only_cropped, BaseGenerator
from prepare_data import ART_WIZARDS as ART

BATCH_SIZE = 64
CODE_SIZE = 100
WIDTH_S = 4
HEIGHT_S = 4
WIDTH = WIDTH_S*2**4
HEIGHT = HEIGHT_S*2**4
LEARNING_RATE = 2e-4

def _gen_res_layer(prev_layer, size, training=True, name='gen_res_layer'):
    with tf.variable_scope(name):
        skip = tf.image.resize_nearest_neighbor(prev_layer, tf.shape(prev_layer)[1:3]*2)
        skip = tf.layers.conv2d(skip, size, 1)
        prev_layer = tf.nn.leaky_relu(prev_layer)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, size, 1, 2, 'same')
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.leaky_relu)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same')
        return skip + prev_layer

def _generator(data, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        size = 64
        prev_layer = tf.layers.dense(data, HEIGHT_S*WIDTH_S*size*2)
        prev_layer = tf.reshape(prev_layer, (-1, HEIGHT_S, WIDTH_S, size*2))
        prev_layer = _gen_res_layer(prev_layer, size*8, training, 'layer0')
        prev_layer = _gen_res_layer(prev_layer, size*4, training, 'layer1')
        prev_layer = _gen_res_layer(prev_layer, size*2, training, 'layer2')
        prev_layer = _gen_res_layer(prev_layer, size*1, training, 'layer3')
        prev_layer = tf.nn.leaky_relu(prev_layer)
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 3, 1, 1, 'same', activation=tf.nn.tanh)
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()


def _dis_res_layer(prev_layer, size, training=True, name='dis_res_layer'):
    with tf.variable_scope(name):
        skip = tf.layers.average_pooling2d(prev_layer, 2, 2)
        skip = tf.layers.conv2d(skip, size, 1)
        prev_layer = tf.nn.leaky_relu(prev_layer)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same', activation=tf.nn.leaky_relu)
        prev_layer = tf.contrib.layers.layer_norm(prev_layer)
        prev_layer = tf.layers.conv2d(prev_layer, size, 3, 1, 'same')
        prev_layer = tf.layers.average_pooling2d(prev_layer, 2, 2)
        return skip + prev_layer

def _discriminator(data, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        size = 64
        prev_layer = tf.layers.conv2d(data, size, 1, 1)
        prev_layer = _dis_res_layer(prev_layer, size*1, training, 'layer0')
        prev_layer = _dis_res_layer(prev_layer, size*2, training, 'layer1')
        prev_layer = _dis_res_layer(prev_layer, size*4, training, 'layer2')
        prev_layer = _dis_res_layer(prev_layer, size*8, training, 'layer3')
        prev_layer = tf.layers.flatten(prev_layer)
        prev_layer = tf.layers.dense(prev_layer, 1, name='logits_wgan')
        if reuse:
            return prev_layer
        else:
            return prev_layer, scope.trainable_variables()

class Generator(BaseGenerator):
    """
        Small Gan Neural Network
    """

    def __init__(self, training=True, batch_size=BATCH_SIZE):
        super().__init__('wizard-gan')
        self.distance = tf.get_variable('distance', [], tf.float32, trainable=False, initializer=tf.initializers.ones)
        #generated
        self.seed = tf.random_uniform((batch_size, CODE_SIZE), -1.0, 1.0)
        self.generated_image, gen_vars = _generator(self.seed, False, training)
        self.generator_critic, disc_vars = _discriminator(self.generated_image, False, training)
        #real
        self.real_image = tf.image.resize_bilinear(get_art_only_cropped(art_list=ART, batch_size=batch_size), (HEIGHT, WIDTH))
        self.real_critic = _discriminator(self.real_image, True, training)
        #fake
        num = tf.cast((batch_size * 0.75 * tf.maximum(0.0, tf.minimum(4.0, self.distance)) + batch_size)/4.0, tf.int32)
        lerp = tf.random_uniform([num, 1, 1, 1], 0.0, 0.7)
        f_img = (1.0-lerp) * self.generated_image[:num, :, :, :] + lerp * self.real_image[:num, :, :, :]
        self.fake_image = tf.concat([self.generated_image[num:, :, :, :], f_img], 0)
        self.fake_critic = _discriminator(self.fake_image, True, training)
        #gp
        self.gp_image = self.real_image + tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0) * (self.generated_image - self.real_image)
        self.gp_critic = _discriminator(self.gp_image, True, training)
        gradients = tf.gradients(self.gp_critic, self.gp_image)
        slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean(tf.square(slope-1.0), name='gradient_penalty')
        #losses
        rmg = tf.reduce_mean(self.generator_critic)
        rmf = tf.reduce_mean(self.fake_critic)
        rmr = tf.reduce_mean(self.real_critic)
        diff = rmr - rmg
        center = tf.reduce_mean(tf.square(self.real_critic))
        self.loss_g = tf.losses.compute_weighted_loss([rmg], [-1.0])
        self.loss_d = tf.losses.compute_weighted_loss(
            [rmf, rmr, self.gradient_penalty, center],
            [1.0, -1.0, 10.0, 0.001])
        loss_d2 = tf.losses.compute_weighted_loss(
            [rmg, rmr, self.gradient_penalty, center],
            [1.0, -1.0, 10.0, 0.001])
        distance = tf.assign(self.distance, 0.9*self.distance + 0.1*diff)
        #training
        learning_rate = tf.square(tf.maximum(0.3, tf.minimum(1.0, distance*2.0))) * LEARNING_RATE
        adam_d = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamD')
        adam_g = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamG')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.trainer_d = adam_d.minimize(self.loss_d, var_list=disc_vars, name='train_d')
            trainer_d2 = adam_d.minimize(loss_d2, var_list=disc_vars, name='train_d2')
            trainer_g = adam_g.minimize(self.loss_g, global_step=self.global_step, var_list=gen_vars, name='train_g')
            self.trainer_g = tf.group(trainer_d2, trainer_g, distance)
        #summary
        tf.summary.scalar("Distance", distance)
        tf.summary.scalar("GradientPenalty", self.gradient_penalty)
        tf.summary.scalar("LossD", self.loss_d)
        tf.summary.scalar("LossG", self.loss_g)
        tf.summary.image("Generated", self.generated_image, 4)

    def train_step(self, summary=False, session=None):
        """
            Do a training step
        """
        if session is None:
            session = self.session
        session.run(self.trainer_d)
        session.run(self.trainer_d)
        if summary:
            _, smry, step, res = session.run([self.trainer_g, self.summary, self.global_step, self.distance])
            self.add_summary(smry, step)
        else:
            _, step, res = session.run([self.trainer_g, self.global_step, self.distance])
        return step, res
