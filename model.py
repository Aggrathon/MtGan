"""
    Tensorflow GAN model
"""
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

DIRECTORY = 'network'
IMAGE_LIST = Path('data') / 'images.txt'
ART_LIST = Path('data') / 'art.txt'
IMAGE_WIDTH = 176
IMAGE_HEIGHT = 128
BATCH_SIZE = 32
CODE_SIZE = 400
LAYERS = 4
LEARNING_RATE = 1e-5

def _read_image(filename):
    string = tf.read_file(filename)
    decoded = tf.image.decode_image(string, 3)
    cropped = tf.image.resize_image_with_crop_or_pad(decoded, IMAGE_HEIGHT, IMAGE_WIDTH)
    factor = 1.0 / 127.5
    floated = tf.to_float(cropped)*factor-1.0
    return floated

def get_image_only_data(list_file=IMAGE_LIST, batch_size=BATCH_SIZE):
    """
        Returns a dataset containing only images
    """
    with tf.device('/cpu:0'):
        textfile = tf.data.TextLineDataset(str(list_file))
        shuffled = textfile.cache().repeat().shuffle(30000)
    images = shuffled.map(_read_image, 8)
    batch = images.batch(batch_size).make_one_shot_iterator().get_next()
    return tf.reshape(batch, (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

#region BEGAN

def _encoder(image, layers, bottleneck=CODE_SIZE):
    layer = image
    shortcuts = []
    for i in range(layers*2):
        size = 32 * (i//2+1)
        if i%2 == 1 and i < layers:
            layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.conv2d(layer, size, 3, i%2+1, 'same', activation=tf.nn.elu, name='conv2d_enc_%d'%i)
        if i%2 == 0 and i > 0:
            sc = tf.layers.conv2d(layer, 4, 3, 2, 'same', activation=tf.nn.elu, name='shortcut_enc_%d_0'%i)
            sc = tf.layers.flatten(sc)
            sc = tf.layers.dense(sc, bottleneck, activation=tf.nn.elu, name='shortcut_enc_%d_1'%i)
            shortcuts.append(sc)
    layer = tf.layers.flatten(layer)
    shortcuts.append(tf.layers.dense(layer, bottleneck, activation=tf.nn.elu))
    layer = tf.concat(shortcuts, -1)
    layer = tf.layers.dense(layer, bottleneck, name='dense_enc')
    return layer

def _decoder(code, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, layers=5):
    w = width // (1<<layers)
    h = height // (1<<layers)
    size = 64
    layer = tf.layers.dense(code, w*h*size, name='dense_dec')
    layer = tf.reshape(layer, (-1, h, w, size))
    first_layer = layer
    for i in range(layers*2+2):
        layer = tf.layers.conv2d(layer, size, 3, 1, 'same', activation=tf.nn.elu, name='conv2d_dec_%d'%i)
        if i == layers*2 + 1:
            layer = tf.layers.conv2d(layer, 3, 3, 1, 'same', name='conv2d_dec_final')
        elif i%2 == 1:
            dbl = 1<<(i//2+1)
            layer = tf.concat([ #upscale and add shortcut
                tf.layers.conv2d_transpose(layer, size, 3, 2, 'same', name='resize_dec_%d'%(i//2)),
                tf.layers.conv2d_transpose(first_layer, size//2, dbl+1, dbl, 'same', name='shortcut_dec_%d'%(i//2))
            ], -1, name='resize_dec_%d_concat'%i)
    return layer

def began_generator(input_size=CODE_SIZE, batch_size=BATCH_SIZE, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, layers=LAYERS):
    """
        The generator neural network for the BEGAN architechture
    """
    with tf.variable_scope('generator'):
        #Generate random input
        prev_layer = tf.random_uniform((batch_size, input_size), -1.0, 1.0)
        #generator == decoder
        prev_layer = _decoder(prev_layer, image_width, image_height, layers)
        return prev_layer

def began_discriminator(image, code_size=CODE_SIZE, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, layers=LAYERS):
    """
        The discriminator neural network for the BEGAN architechture
    """
    with tf.variable_scope('discriminator'):
        #autoencoder
        prev_layer = _encoder(image, layers, code_size)
        prev_layer = _decoder(prev_layer, image_width, image_height, layers)
        return prev_layer

def began_trainer(real_image, fake_image, real_result, fake_result, learning_rate=LEARNING_RATE):
    """
        The trainer for the BEGAN
    """
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('trainer'):
        # losses
        L_x = tf.reduce_mean(tf.abs(real_image-real_result), name="L_x")
        L_g = tf.reduce_mean(tf.abs(fake_image-fake_result), name="L_G")
        # hyperparameters
        gamma = 0.15 + 0.25*tf.to_float(tf.minimum(global_step, 100000))/100000.0    # higher == more diversity, lower == better quality
        lmbd = 0.001    # learning rate for k (and k tries to keep L_g /L_x = gamma)
        k = tf.Variable(0.01, False, name='kt')
        k = tf.assign(k, tf.maximum(1e-5, k + lmbd*(gamma*L_x-L_g)), name='kt1')
        #training
        L_D = tf.subtract(L_x, k*L_g, name='L_D')
        L_G = L_g
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_d = optimizer.minimize(L_D, global_step, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
            train_op_g = optimizer.minimize(L_G, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        #summary
        measure = tf.add(L_x, tf.abs(gamma*L_x-L_g), 'measure')
        tf.summary.scalar("Measure", measure)
        tf.summary.scalar("kt", k)
        tf.summary.scalar("LossD", L_D)
        tf.summary.scalar("LossG", L_g)
        tf.summary.scalar("LossX", L_x)
        tf.summary.image("Generated", fake_image, 4)
        tf.summary.image("RealReconstruct", real_result, 1)
        tf.summary.image("FakeReconstruct", fake_result, 1)
        return tf.group(k, train_op_d, train_op_g), measure

#endregion BEGAN


#region WGAN-GP

def wgangp_generator(input_size=CODE_SIZE, batch_size=BATCH_SIZE, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, layers=LAYERS):
    """
        The generator neural network for the WGAN-GP architechture
    """
    with tf.variable_scope('generator') as scope:
        #Generate random input
        first_layer = tf.random_uniform((batch_size, input_size), -1.0, 1.0)
        image_width = image_width // (1<<layers)
        image_height = image_height // (1<<layers)
        sizes = [256, 128, 64, 32, 16]
        layer = tf.layers.dense(first_layer, image_height*image_width*sizes[1], name='dense')
        layer = tf.reshape(layer, (batch_size, image_height, image_width, sizes[1]))
        for i in range(layers+1):
            if i <= layers//2:
                layer = tf.layers.batch_normalization(layer, name='batch_norm_%d'%i)
            layer = tf.layers.conv2d(layer, sizes[i], 3, 1, 'same', activation=tf.nn.elu, name='conv2d_%d'%i)
            if i == layers:
                layer = tf.layers.conv2d(layer, 3, 3, 1, 'same', activation=tf.nn.tanh, name='conv2d_final')
            else:
                image_width *= 2
                image_height *= 2
                shortcut = tf.layers.dense(first_layer, image_height*image_width*sizes[i+1], name='shortcut_%d'%i)
                shortcut = tf.reshape(shortcut, (batch_size, image_height, image_width, sizes[i+1]))
                layer = tf.concat([ #upscale and add shortcut
                    tf.layers.conv2d_transpose(layer, sizes[i], 3, 2, 'same', name='resize_%d'%i),
                    shortcut
                ], -1, name='resize_concat_%d'%i)
        return layer, scope.trainable_variables()

def wgangp_discriminator(images, layers=LAYERS):
    """
        The discriminator neural network for the WPGAN-GP architechture
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        layer = images
        sizes = [32, 64, 128, 256]
        outputs = []
        for layer in images:
            shortcuts = []
            for i in range(layers):
                if i <= layers//2:
                    with tf.variable_scope('layer_norm_%d'%i, reuse=tf.AUTO_REUSE) as s:
                        layer = tf.contrib.layers.layer_norm(layer, scope=s)
                layer = tf.layers.conv2d(layer, sizes[i], 3, 1, 'same', activation=tf.nn.elu, name='conv2d_%d'%i)
                layer = tf.layers.conv2d(layer, sizes[i], 3, 2, 'same', activation=tf.nn.elu, name='resize_%d'%i)
                sc = tf.layers.conv2d(layer, 4, 3, 1, 'same', activation=tf.nn.elu, name='shortcut_%d_0'%i)
                sc = tf.layers.flatten(sc)
                sc = tf.layers.dense(sc, 64, activation=tf.nn.elu, name='shortcut_%d_1'%i)
                shortcuts.append(sc)
            layer = tf.concat(shortcuts, -1)
            layer = tf.layers.dense(layer, 128, activation=tf.nn.elu, name='dense_0')
            layer = tf.layers.dense(layer, 32, activation=tf.nn.elu, name='dense_1')
            layer = tf.layers.dense(layer, 1, name='logits')
            outputs.append(layer)
            scope.reuse_variables()
        return outputs, scope.trainable_variables()

def wgangp_trainer(real_g, fake_g, hat_g, real_d, fake_d, hat_d, vars_g, vars_d, learning_rate=LEARNING_RATE):
    """
        The trainer for the WGAN-GP
    """
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('trainer'):
        #Losses
        L_G = -tf.reduce_mean(fake_d)*0.5
        slope = tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(hat_d, hat_g, colocate_gradients_with_ops=True)[0]), -1))
        gradient_penalty = tf.reduce_mean(tf.square(slope-1.0))
        L_D = tf.reduce_mean(fake_d) - tf.reduce_mean(real_d) + 10*gradient_penalty
        #Trainers
        trainer_d = tf.train.AdamOptimizer(learning_rate, 0.5, 0.9, name='AdamD')
        trainer_g = tf.train.AdamOptimizer(learning_rate, 0.5, 0.9, name='AdamG')
        op_d = trainer_d.minimize(L_D, global_step, vars_d, name='trainD', colocate_gradients_with_ops=True)
        op_g = trainer_g.minimize(L_G, None, vars_g, name='trainG', colocate_gradients_with_ops=True)
        train_op = tf.group(op_d, op_g)
        #summary
        measure = tf.get_variable('measure', [], tf.float32, trainable=False)
        measure = tf.assign(measure, 0.9*measure-0.1*L_D)
        tf.summary.scalar("Measure", measure)
        tf.summary.scalar("GradientPenalty", gradient_penalty)
        tf.summary.scalar("LossD", L_D)
        tf.summary.scalar("LossG", L_G)
        tf.summary.image("Generated", fake_g, 4)
        return tf.group(op_d, op_g, measure), measure


#endregion WGAN-GP


class Generator():
    def __init__(self, name):
        self.name = name
    def train_step(self, session, summary=None):
        """
            Do a training step and return step_nr and result (and optionally summary to write)
        """
        pass


class ArtGeneratorBEGAN(Generator):
    def __init__(self):
        super().__init__('began')
        with tf.variable_scope(self.name):
            os.makedirs(DIRECTORY, exist_ok=True)
            with tf.device('/cpu:0'):
                    self.learning_rate = tf.Variable(LEARNING_RATE, False, dtype=tf.float32, name='learning_rate')
                    self.best_step = tf.Variable(0, False, dtype=tf.int32, name='best_step')
                    self.best_result = tf.Variable(999.0, False, dtype=tf.float32, name='best_result')
                    tf.summary.scalar("LearningRate", self.learning_rate)
            self.global_step = tf.train.get_or_create_global_step()
            self.real_image = get_image_only_data(ART_LIST)
            self.fake_image = began_generator()
            self.real_reconstruct, self.fake_reconstruct = tf.split(began_discriminator(tf.concat([self.real_image, self.fake_image], 0)), 2, 0)
            self.trainer, self.measure = began_trainer(self.real_image, self.fake_image, self.real_reconstruct, self.fake_reconstruct, self.learning_rate)

    def train_step(self, session, summary=None):
        if summary is not None:
            _, smry, step, res, br, bs = session.run([self.trainer, summary, self.global_step, self.measure, self.best_result, self.best_step])
        else:
            _, step, res, br, bs = session.run([self.trainer, self.global_step, self.measure, self.best_result, self.best_step])
        if res < br:
            if bs == step-1:
                lr = self.learning_rate.eval(session)*1.05
                session.run([self.learning_rate.assign(lr), self.best_result.assign(res), self.best_step.assign(step)])
            else:
                session.run([self.best_result.assign(res), self.best_step.assign(step)])
        elif bs + 300 < step:
            limit = np.exp(-(np.log(step/100.0+1000.0)**2)+45.0)*0.001
            lr = max(self.learning_rate.eval(session)*0.8, limit)
            session.run([self.learning_rate.assign(lr), self.best_result.assign(res), self.best_step.assign(step)])
        if summary is not None:
            return step, res, smry
        else:
            return step, res

class ArtGeneratorWGANGP(Generator):
    def __init__(self):
        super().__init__('wgan-gp')
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(self.name):
            os.makedirs(DIRECTORY, exist_ok=True)
            self.real_image = get_image_only_data(ART_LIST)
            self.fake_image, self.generator_variables = wgangp_generator()
            epsilon = tf.random_uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
            self.hat_image = epsilon*self.real_image + (1.0-epsilon)*self.real_image
            imgs = [self.real_image, self.fake_image, self.hat_image]
            discs, self.discriminator_variables = wgangp_discriminator(imgs)
            self.real_disc, self.fake_disc, self.hat_disc = discs
            self.trainer, self.measure = wgangp_trainer( \
                self.real_image, self.fake_image, self.hat_image, \
                self.real_disc, self.fake_disc, self.hat_disc, \
                self.generator_variables, self.discriminator_variables)

    def train_step(self, session, summary=None):
        if summary is not None:
            _, smry, step, res = session.run([self.trainer, summary, self.global_step, self.measure])
            return step, res, smry
        else:
            _, step, res = session.run([self.trainer, self.global_step, self.measure])
            return step, res
