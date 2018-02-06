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
    rd, fd = tf.split(discriminator(tf.concat([ri, fi], 0)), 2, 0)
    train_op = trainer(ri, fi, rd, fd)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(DIRECTORY)
    saver = tf.train.Saver()
    global_step = tf.train.get_or_create_global_step()
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True #pylint: disable=E1101
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(DIRECTORY))
        except:
            sess.run(tf.global_variables_initializer())
        print("Starting the training")
        try:
            for i in range(1000):
                _, smry, step = sess.run([train_op, summary, global_step])
                writer.add_summary(smry, step)
                if step == 1:
                    writer.add_graph(sess.graph, step)
                if i%10 == 5:
                    saver.save(sess, os.path.join(DIRECTORY, 'model'), step)
                new_time = timer()
                print('%6d:  \t%.2fs'%(step, new_time-old_time))
                old_time = new_time
                for _ in range(99):
                    sess.run(train_op)
            step = sess.run(global_step)
            saver.save(sess, os.path.join(DIRECTORY, 'model'), step)
        except KeyboardInterrupt:
            step = sess.run(global_step)
            saver.save(sess, os.path.join(DIRECTORY, 'model'), step)



if __name__ == "__main__":
    train()
