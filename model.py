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
LEARNING_RATE = 1e-4

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

def _encoder(image, layers, bottleneck=CODE_SIZE):
    layer = image
    shortcuts = []
    for i in range(layers*2):
        size = 16 * (i//2+2)
        if i%2 == 1 and i < layers:
            layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.conv2d(layer, size, 5, i%2+1, 'same', activation=tf.nn.elu, name='conv2d_enc_%d'%i)
        if i%2 == 0 and i > 0:
            sc = tf.layers.conv2d(layer, 4, 5, 2, 'same', activation=tf.nn.elu, name='shortcut_enc_%d_0'%i)
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
    size = 48
    layer = tf.layers.dense(code, w*h*size, name='dense_dec')
    layer = tf.reshape(layer, (-1, h, w, size))
    first_layer = layer
    for i in range(layers*2+2):
        layer = tf.layers.conv2d(layer, size, 5, 1, 'same', activation=tf.nn.elu, name='conv2d_dec_%d'%i)
        if i == layers*2 + 1:
            layer = tf.layers.conv2d(layer, 3, 3, 1, 'same', name='conv2d_dec_final')
        elif i%2 == 1:
            dbl = 1<<(i//2+1)
            layer = tf.concat([ #upscale and add shortcut
                tf.layers.conv2d_transpose(layer, size, 3, 2, 'same', name='resize_dec_%d'%(i//2)),
                tf.layers.conv2d_transpose(first_layer, size//2, dbl+1, dbl, 'same', name='shortcut_dec_%d'%(i//2))
            ], -1, name='resize_dec_%d_concat'%i)
    return layer

def _add_noise(image, batch_size=BATCH_SIZE, max_amount=0.3, no_noise=0.2):
    #not used
    noise = tf.maximum(tf.random_uniform(batch_size, 0.0, no_noise+max_amount) - no_noise, 0.0)
    image = tf.maximum(0.0, tf.minimum(1.0, image + tf.random_uniform(tf.shape(image), -1.0, 1.0)*noise))
    return image


def generator(input_size=CODE_SIZE, batch_size=BATCH_SIZE, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, layers=LAYERS):
    """
        The generator neural network for the GAN architechture
    """
    with tf.variable_scope('generator'):
        #Generate random input
        prev_layer = tf.random_uniform((batch_size, input_size), -1.0, 1.0)
        #generator == decoder
        prev_layer = _decoder(prev_layer, image_width, image_height, layers)
        return prev_layer

def discriminator(image, code_size=CODE_SIZE, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, layers=LAYERS):
    """
        The discriminator neural network for the GAN architechture
    """
    with tf.variable_scope('discriminator'):
        #autoencoder
        prev_layer = _encoder(image, layers, code_size)
        prev_layer = _decoder(prev_layer, image_width, image_height, layers)
        return prev_layer


def trainer(real_image, fake_image, real_result, fake_result, learning_rate=LEARNING_RATE):
    """
        The trainer for the GAN
    """
    global_step = tf.train.get_or_create_global_step()
    # losses
    L_x = tf.reduce_mean(tf.abs(real_image-real_result), name="L_x")
    L_g = tf.reduce_mean(tf.abs(fake_image-fake_result), name="L_G")
    # hyperparameters
    gamma = 0.2 + 0.4*tf.to_float(tf.minimum(global_step, 100000))/100000.0    # higher == more diversity, lower == better quality
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


class ArtGenerator():
    def __init__(self):
        os.makedirs(DIRECTORY, exist_ok=True)
        with tf.device('/cpu:0'):
            self.learning_rate = tf.Variable(LEARNING_RATE, False, dtype=tf.float32, name='learning_rate')
            self.best_step = tf.Variable(0, False, dtype=tf.int32, name='best_step')
            self.best_result = tf.Variable(999.0, False, dtype=tf.float32, name='best_result')
            tf.summary.scalar("LearningRate", self.learning_rate)
            self.global_step = tf.train.get_or_create_global_step()
        self.real_image = get_image_only_data(ART_LIST)
        self.fake_image = generator()
        self.real_reconstruct, self.fake_reconstruct = tf.split(discriminator(tf.concat([self.real_image, self.fake_image], 0)), 2, 0)
        self.trainer, self.measure = trainer(self.real_image, self.fake_image, self.real_reconstruct, self.fake_reconstruct, self.learning_rate)

    def train_step(self, session, summary=None):
        if summary is not None:
            _, smry, step, res, br, bs = session.run([self.trainer, summary, self.global_step, self.measure, self.best_result, self.best_step])
        else:
            _, step, res, br, bs = session.run([self.trainer, self.global_step, self.measure, self.best_result, self.best_step])
        if res < br:
            if bs == step-1:
                lr = self.learning_rate.eval(session)*1.025
                session.run([self.learning_rate.assign(lr), self.best_result.assign(res), self.best_step.assign(step)])
            else:
                session.run([self.best_result.assign(res), self.best_step.assign(step)])
        elif bs + 300 < step:
            limit = np.power(10.0, -(np.log(step)/2.2+1.5))
            lr = max(self.learning_rate.eval(session)*0.75, limit)
            session.run([self.learning_rate.assign(lr), self.best_result.assign(res), self.best_step.assign(step)])
        if summary is not None:
            return step, res, smry
        else:
            return step, res
