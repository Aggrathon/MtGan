"""
    Run this script to train the neural network
"""
import math
from timeit import default_timer as timer
import tensorflow as tf
#from models.wgan_gp import ArtGeneratorWGANGP as Generator
from models.pg_gan import PgGanGenerator as Generator


def train():
    """
        Train the neural network
    """
    old_time = timer()
    new_time = timer()
    generator = Generator()
    with generator:
        print("Starting the training")
        for i in range(1000):
            step, result = generator.train_step(summary=True)
            if i%100 == 50:
                generator.save()
                print('\t'*6+'saved', end='\r')
            new_time = timer()
            print('%6d:  \t%2.2fs  \t%2.2f'%(step, new_time-old_time, result))
            old_time = new_time
            for _ in range(99):
                step, result = generator.train_step()
                if math.isnan(result):
                    print('Training Failed (%d)'%step)
                    data = generator.session.run((generator.loss_g, generator.loss_d, generator.gradient_penalty))
                    print('Generator Loss: %.2f, Discriminator Loss: %.2f, Gradient Penalty: %.2f'%data)
                    return

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    train()
