"""
    Run this script to train the neural network
"""
import os
from timeit import default_timer as timer
import tensorflow as tf
from model import ArtGenerator, DIRECTORY

MODEL_PATH = os.path.join(DIRECTORY, 'model')

def train():
    """
        Train the neural network
    """
    old_time = timer()
    new_time = timer()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    generator = ArtGenerator()
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
            #sess.run(generator.learning_rate.assign(4e-5))
            for i in range(1000):
                step, result, smry = generator.train_step(sess, summary)
                writer.add_summary(smry, step)
                if step == 1:
                    writer.add_graph(sess.graph, step)
                if i%10 == 5:
                    saver.save(sess, MODEL_PATH, step)
                new_time = timer()
                print('%6d:  \t%.2fs  \t%.2f'%(step, new_time-old_time, result))
                old_time = new_time
                for _ in range(99):
                    generator.train_step(sess)
            saver.save(sess, MODEL_PATH, global_step.eval(sess))
        except KeyboardInterrupt:
            print("Stopping the training")
            saver.save(sess, MODEL_PATH, global_step.eval(sess))



if __name__ == "__main__":
    train()
