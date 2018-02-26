"""
    The PG-GAN neural network model
	https://arxiv.org/abs/1710.10196
"""
import tensorflow as tf
from models.model import IMAGE_HEIGHT, IMAGE_WIDTH, get_art_only_data, Generator

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
ITERATIONS_PER_LOD = 20000

def _generator_layer(prev_layer, images, filters, lerp, name):
	with tf.variable_scope(name):
		out = tf.image.resize_bilinear(images[-1], tf.multiply(tf.shape(prev_layer)[1:3], 2))
		prev_layer = tf.layers.conv2d_transpose(prev_layer, filters, 3, 2, 'same', activation=tf.nn.leaky_relu, name='Conv0')
		prev_layer = tf.layers.conv2d(prev_layer, filters, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Conv1')
		image = tf.layers.conv2d(prev_layer, 3, 3, 1, 'same', name='Output%d'%(len(images)+1))
		images.append(tf.add(out*(1.0-lerp), image*lerp, name='Output%d'%len(images)))
		images.append(image)
		return prev_layer

def _discriminator_layer(image, filters, first, last, prev, lerp, name):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		if first:
			layer = tf.layers.conv2d(image, filters//2, 1, 1, 'same', activation=tf.nn.leaky_relu, name='ConvInit')
		elif lerp is not None and prev is not None:
			layer = tf.image.resize_bilinear(prev, tf.shape(prev)[1:3]//2)
			layer = tf.layers.conv2d(layer, filters//2, 1, 1, 'same', activation=tf.nn.leaky_relu, name='ConvInit')
			layer = layer*(1.0-lerp) + image*lerp
		else:
			layer = image
		layer = tf.layers.conv2d(layer, filters, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Conv0')
		if last:
			layer = tf.layers.conv2d(layer, filters, 3, 1, 'valid', activation=tf.nn.leaky_relu, name='Conv1')
			layer = tf.layers.dense(layer, filters, tf.nn.leaky_relu, name='Dense0')
			layer = tf.layers.dense(layer, 1, name='Dense1')
		else:
			layer = tf.layers.conv2d(layer, filters, 3, 2, 'same', activation=tf.nn.leaky_relu, name='Conv1')
		return layer

def _discriminator(image, layer=None, prev=None, level=0, lerp=None):
	if level < 0:
		return layer
	size = [512, 256, 128, 64, 32][level]
	if image is None:
		layer = _discriminator_layer(layer, size, False, level == 0, prev, lerp, 'Layer%d'%level)
		return _discriminator(None, layer, None, level-1, lerp)
	else:
		layer = _discriminator_layer(image, size, True, level == 0, prev, lerp, 'Layer%d'%level)
		return _discriminator(None, layer, image, level-1, lerp)


class PgGanGenerator(Generator):
	"""
        Neural network for generating art based on the PG-GAN architechture
	"""

	def __init__(self, batch_size=BATCH_SIZE, iterations_per_lod=ITERATIONS_PER_LOD, learning_rate=LEARNING_RATE):
		super().__init__("pg-gan")
		with self.scope:
			self.seed = tf.random_uniform((batch_size, 1, 1, 512))
			self.lerp = tf.to_float(tf.mod(self.global_step, iterations_per_lod)) / tf.to_float(iterations_per_lod)
			self.fake_images = []
			with tf.variable_scope('Generator') as scope:
				with tf.variable_scope('Layer0'):
					layer = tf.layers.conv2d_transpose(self.seed, 512, (8, 11), 1, activation=tf.nn.leaky_relu, name='Conv0')
					layer = tf.layers.conv2d(layer, 512, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Conv1')
					self.fake_images.append(tf.layers.conv2d(layer, 3, 3, 1, 'same', name='Output%d'%len(self.fake_images)))
				layer = _generator_layer(layer, self.fake_images, 256, self.lerp, 'Layer1')
				layer = _generator_layer(layer, self.fake_images, 128, self.lerp, 'Layer2')
				layer = _generator_layer(layer, self.fake_images, 64, self.lerp, 'Layer3')
				layer = _generator_layer(layer, self.fake_images, 32, self.lerp, 'Layer4')
				self.generator_variables = scope.trainable_variables()
			self.fake_discs = []
			self.real_discs = []
			self.hat_discs = []
			self.real_image = get_art_only_data(batch_size=batch_size)
			self.real_images = []
			self.hat_images = []
			with tf.variable_scope('Discriminator') as scope:
				for i, image in enumerate(self.fake_images):
					layers = (i+1)%2
					mult = 1 << ((i+1)//2)
					real_img = tf.image.resize_bilinear(self.real_image, (8*mult, 11*mult))
					rnd = tf.random_uniform((batch_size, 1, 1, 1), 0.0, 1.0)
					hat_img = real_img + (image - real_img)*rnd
					self.real_images.append(real_img)
					self.hat_images.append(hat_img)
					if i%2 == 1:
						self.real_discs.append(_discriminator(real_img, None, None, layers, self.lerp))
						self.fake_discs.append(_discriminator(image, None, None, layers, self.lerp))
						self.hat_discs.append(_discriminator(hat_img, None, None, layers, self.lerp))
					else:
						self.real_discs.append(_discriminator(real_img, None, None, layers, None))
						self.fake_discs.append(_discriminator(image, None, None, layers, None))
						self.hat_discs.append(_discriminator(hat_img, None, None, layers, None))
				self.discriminator_variables = scope.trainable_variables()
			with tf.variable_scope('Loss'):
				self.losses_g = []
				self.losses_d = []
				self.gradient_penalties = []
				for real_img, fake_img, hat_img, real_d, fake_d, hat_d in\
					zip(self.real_images, self.fake_images, self.hat_images, self.real_discs, self.fake_discs, self.hat_discs):
					self.losses_d.append(-tf.reduce_mean(fake_d))
					gradients = tf.gradients(hat_d, hat_img)
					slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
					gp = tf.reduce_mean(tf.square(slope-1.0))
					self.gradient_penalties.append(gp)
					self.losses_g.append(tf.losses.compute_weighted_loss(
						[tf.reduce_mean(fake_d), tf.reduce_mean(real_d), gp, tf.reduce_mean(tf.square(fake_d))],
						[1.0, -1.0, 10.0, 0.001]
					))
			with tf.variable_scope('Trainer'):
				self.trainer_g = []
				self.trainer_d = []
				adam_d = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamD')
				adam_g = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamG')
				with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
					for ld, lg in zip(self.losses_d, self.losses_g):
						self.trainer_d.append(adam_d.minimize(ld, var_list=self.discriminator_variables))
						self.trainer_g.append(adam_g.minimize(lg, global_step=self.global_step, var_list=self.generator_variables))
			with tf.variable_scope('Summary'):
				self.measure = tf.get_variable('Measure', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.gradient_penalty = tf.get_variable('GradientPenalty', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.loss_d = tf.get_variable('LossD', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.loss_g = tf.get_variable('LossG', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.generated = tf.get_variable('Generated', [4, IMAGE_HEIGHT, IMAGE_WIDTH, 3], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.summaries = []
				for ld, lg, gp, fi in zip(self.losses_d, self.losses_g, self.gradient_penalties, self.fake_images):
					self.summaries.append(tf.group(
						tf.assign(self.measure, 0.95*self.measure-0.05*ld),
						tf.assign(self.loss_d, ld),
						tf.assign(self.loss_g, lg),
						tf.assign(self.generated, tf.image.resize_nearest_neighbor(fi[:4,:,:,:], [IMAGE_HEIGHT, IMAGE_WIDTH]))
					))
			tf.summary.scalar("Measure", self.measure)
			tf.summary.scalar("GradientPenalty", self.gradient_penalty)
			tf.summary.scalar("LossD", self.loss_d)
			tf.summary.scalar("LossG", self.loss_g)
			tf.summary.image("Generated", self.generated, 4)

	def train_step(self, summary=False, session=None):
		"""
			Do a training step
		"""
		if session is None:
			session = self.session
		step = self.global_step.eval(session)
		lod = min(step // ITERATIONS_PER_LOD, 8)
		for _ in range(2):
			session.run(self.trainer_d[lod], dict(self.lerp))
		if summary:
			_, _, step, res = session.run([self.trainer_g[lod], self.summaries[lod], self.global_step, self.measure])
			self.add_summary(session.run(self.summary), step)
		else:
			_, step, res = session.run([self.trainer_g[lod], self.global_step, self.measure])
		return step, res

