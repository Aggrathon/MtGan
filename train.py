"""
    Run this script to train the neural network
"""
import os
from timeit import default_timer as timer
import tensorflow as tf
from model import get_image_only_data, generator, discriminator, trainer, DIRECTORY

def train():
    """
        Train the neural network
    """
    old_time = timer()
    new_time = timer()
    os.makedirs(DIRECTORY, exist_ok=True)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    ri = get_image_only_data()
    fi = generator()
    rd = discriminator(ri, False)
    fd = discriminator(fi, True)
    train_op = trainer(ri, fi, rd, fd)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(DIRECTORY)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint ignore=E1101
    saver = tf.train.Saver()
    global_step = tf.train.get_or_create_global_step()
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(DIRECTORY))
        except:
            sess.run(tf.global_variables_initializer())
        print("Starting the training")
        for i in range(1000):
            _, smry, step = sess.run([train_op, summary, global_step])
            writer.add_summary(smry, step)
            if i%10 == 5:
                saver.save(sess, os.path.join(DIRECTORY, 'model'), step)
            new_time = timer()
            print('%d:\t%.2fs'%(step, new_time-old_time))
            old_time = new_time
            for _ in range(99):
                sess.run(train_op)
        saver.save(sess, os.path.join(DIRECTORY, 'model'), step)



if __name__ == "__main__":
    train()
