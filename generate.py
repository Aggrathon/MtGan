"""
    Generate images
"""
import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from models.specific_gan import Generator, WIDTH, HEIGHT, CODE_SIZE

def _latent_seed(grid_size, generator):
    index = 0
    seed, scores = generator.session.run([generator.seed, generator.generator_critic])
    for i in range(1, 5):
        x = np.argmax(scores[:-i])
        print(x, scores)
        seed[-i] = seed[x]
        scores[x] -= 1000
    for i in range(grid_size):
        for j in range(grid_size):
            seed[index] = \
                seed[-1] * (1.0 - i/grid_size) * (1.0 - j/grid_size) + \
                seed[-2] * (i/grid_size) * (1.0 - j/grid_size) + \
                seed[-3] * (1.0 - i/grid_size) * (j/grid_size) + \
                seed[-4] * (i/grid_size) * (j/grid_size)
            index += 1
    return seed

def grid(latent=False, number=1):
    grid_size = 5
    generator = Generator()
    image = tf.contrib.gan.eval.image_grid( \
        generator.generated_image[:grid_size*grid_size, :, :, :],
        [grid_size, grid_size],
        [HEIGHT, WIDTH], 3)
    image = image[0] * 0.5 + 0.5
    with generator:
        for _ in range(number):
            if latent:
                img = generator.session.run(image, feed_dict={generator.seed: _latent_seed(grid_size, generator)})
            else:
                img = generator.session.run(image)
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'help':
        print("Usage:")
        print("\tpython %s [latent] [num_grids]"%sys.argv[1])
        print("Using 'latent' causes the the seeds to be sampled")
        print("from latent vector space instead of randomly")
    if len(sys.argv) > 1 and sys.argv[1] == 'latent':
        if len(sys.argv) == 3:
            grid(True, int(sys.argv[2]))
        else:
            grid(True, 1)
    else:
        if len(sys.argv) == 2:
            grid(False, int(sys.argv[2]))
        else:
            grid(False, 1)
