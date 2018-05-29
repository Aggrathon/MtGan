"""
    Run this script to train the neural network
"""
import math
from timeit import default_timer as timer
import tensorflow as tf
#from models.wgan_gp import ArtGeneratorWGANGP as Generator
#from models.pg_gan import PgGanGenerator as Generator
#from models.skip_gan import SkipGanGenerator as Generator
#from models.small_gan import SmallGanGenerator as Generator
#from models.residual_gan import ResidualGenerator as Generator
#from models.varied2_gan import VariedGenerator as Generator
#from models.specific_gan import Generator
from models.wizard_gan import Generator


def train():
    """
        Train the neural network
    """
    start_time = timer()
    next_time = start_time
    generator = Generator()
    with generator:
        print("Starting the training")
        step, result = generator.train_step()
        time = timer() - start_time
        print("Iteration    Time     Distance")
        #      123456:123456:12:12:123456.89
        print('%6d:%6d:%02d:%02d%9.2f'%(step, time//3600, (time//60)%60, int(time)%60, result))
        for i in range(6000): # 60 * 6000 + ish = 100h + ish
            next_time = timer() + 60.0
            while timer() < next_time:
                step, result = generator.train_step()
                if math.isnan(result):
                    print('Training Failed (%d)'%step)
                    data = generator.session.run((generator.loss_g, generator.loss_d, generator.gradient_penalty))
                    print('Generator Loss: %.2f, Discriminator Loss: %.2f, Gradient Penalty: %.2f'%data)
                    return
            step, result = generator.train_step(summary=True)
            if i%10 == 9:
                generator.save()
                print('                                saved', end='\r')
                #      123456:123456:12:12:123456.89
            time = timer() - start_time
            print('%6d:%6d:%02d:%02d%9.2f'%(step, time//3600, (time//60)%60, int(time)%60, result))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    train()
