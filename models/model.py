"""
    Tensorflow GAN model
"""
import os
from pathlib import Path
import tensorflow as tf

DIRECTORY = Path('network')
IMAGE_LIST = Path('data') / 'images.txt'
ART_LIST = Path('data') / 'art.txt'
IMAGE_WIDTH = 176
IMAGE_HEIGHT = 128
GPU = False

def _read_image(filename):
    string = tf.read_file(filename)
    decoded = tf.image.decode_image(string, 3)
    cropped = tf.image.resize_image_with_crop_or_pad(decoded, IMAGE_HEIGHT, IMAGE_WIDTH)
    factor = 1.0 / 127.5
    floated = tf.to_float(cropped)*factor-1.0
    return floated

def _read_image_random(filename):
    image = _read_image(filename)
    random = tf.image.random_brightness(image, 0.2)
    random = tf.image.random_contrast(random, 0.85, 1.2)
    random = tf.image.random_flip_left_right(random)
    random = tf.image.random_saturation(random, 0.85, 1.2)
    random = tf.image.random_hue(random, 0.05)
    return random

def get_image_only_data(list_file=IMAGE_LIST, batch_size=32):
    """
        Returns a dataset containing only images
    """
    with tf.device('/cpu:0'):
        textfile = tf.data.TextLineDataset(str(list_file))
        shuffled = textfile.cache().repeat().shuffle(30000)
    images = shuffled.map(_read_image, 8)
    batch = images.batch(batch_size).make_one_shot_iterator().get_next()
    return tf.reshape(batch, (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

def get_art_only_data(list_file=ART_LIST, batch_size=32):
    """
        Returns a dataset containing only art
    """
    with tf.device('/cpu:0'):
        textfile = tf.data.TextLineDataset(str(list_file))
        shuffled = textfile.cache().repeat().shuffle(30000)
    images = shuffled.map(_read_image_random, 8)
    batch = images.batch(batch_size).make_one_shot_iterator().get_next()
    return tf.reshape(batch, (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

class Generator():
    """
        Base class for all generators
    """
    def __init__(self, name):
        self.name = name
        os.makedirs(str(DIRECTORY / name), exist_ok=True)
        self.session = None
        self.saver = None
        self.summary = None
        self.global_step = tf.train.get_or_create_global_step()
        self.scope = tf.variable_scope(name)
        self.summary_writer = None

    def restore(self, session=None):
        """
            Restore a saved model or initialize a new one
        """
        if session is None: session = self.session
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(str(DIRECTORY / self.name))
        try:
            self.saver.restore(session, tf.train.latest_checkpoint(str(DIRECTORY / self.name)))
            print("Loaded an existing model")
        except:
            session.run(tf.global_variables_initializer())
            self.summary_writer.add_graph(session.graph, 0)
            print("Created a new model")

    def save(self, session=None):
        """
            Save the current model
        """
        if session is None: session = self.session
        self.saver.save(session, str(DIRECTORY / self.name / 'model'), self.global_step)

    def train_step(self, summary=False, session=None):
        """
            Do a training step and return step_nr and result (and optionally summary to write)
        """
        if session is None:
            session = self.session
    
    def add_summary(self, event, step):
        """
            Write a tensorboard summary
        """
        self.summary_writer.add_summary(event, step)

    def __enter__(self):
        print("Initializing Tensorflow Session")
        if GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True #pylint: disable=E1101
            self.session = tf.Session(config=config)
        else:
            self.session = tf.Session()
        self.restore()
        return self

    def __exit__(self, type_, value, traceback):
        if self.global_step.eval(self.session) > 100:
            self.save()
        self.session.close()
