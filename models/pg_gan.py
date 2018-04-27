"""
    The PG-GAN neural network model
	https://arxiv.org/abs/1710.10196
"""
import tensorflow as tf
from models.model import get_art_only_cropped, Generator, IMAGE_HEIGHT, IMAGE_WIDTH

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
ITERATIONS_PER_LOD = 10000

GENERATOR_WIDTHS = [512, 512, 384, 256]
DISCRIMINATOR_WIDTHS = [512, 384, 256, 192, 128]

def _lerp(lerp, a, b):
	return a + (b-a)*lerp

def _generator(data, depth, reuse=False, training=True):
	if depth > 0:
		layer, _ = _generator(data, depth-1, True)
	filters = GENERATOR_WIDTHS[depth]
	with tf.variable_scope('Generator%d'%depth, reuse=reuse):
		if depth == 0:
			layer = tf.layers.conv2d_transpose(data, 256, (8, 11), 1, name='Expand')
		prev_layer = tf.layers.conv2d_transpose(layer, filters, 3, 2, 'same', activation=tf.nn.leaky_relu, name='Conv0')
		prev_layer = tf.layers.conv2d(prev_layer, filters*2, 1, 1, 'same', name='Conv1')
		prev_layer = tf.layers.conv2d(prev_layer, filters, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Conv2')
		prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
		image = tf.layers.conv2d(prev_layer, 3, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Output0')
		image = tf.layers.conv2d(image, 3, 3, 1, 'same', activation=tf.nn.tanh, name='Output1')
		return prev_layer, image

def _discriminator(image, depth, reuse=False, training=True):
	filters = DISCRIMINATOR_WIDTHS[depth]
	with tf.variable_scope('Discriminator%d'%depth, reuse=reuse):
		if int(image.get_shape()[-1]) == 3:
			layer = tf.layers.conv2d(image, DISCRIMINATOR_WIDTHS[depth+1], 3, 1, 'same', activation=tf.nn.leaky_relu, name='Input')
		else:
			layer = image
		layer = tf.layers.conv2d(layer, filters, 3, 1, 'same', activation=tf.nn.leaky_relu, name='Conv0')
		layer = tf.layers.conv2d(layer, filters*2, 1, 1, 'same', name='Conv1')
		layer = tf.layers.conv2d(layer, filters, 3, 2, 'same', activation=tf.nn.leaky_relu, name='Conv2')
		if depth == 0:
			layer = tf.layers.conv2d(layer, 512, 1, 1, 'same', activation=tf.nn.leaky_relu, name='Output0')
			layer = tf.layers.conv2d(layer, 256, 3, 1, 'valid', activation=tf.nn.leaky_relu, name='Output1')
			layer = tf.layers.flatten(layer)
			return tf.layers.dense(layer, 1, name='logits')
		else:
			with tf.variable_scope('layer_norm', reuse=tf.AUTO_REUSE) as norm_scope:
				layer = tf.contrib.layers.layer_norm(layer, begin_norm_axis=1, begin_params_axis=1, scope=norm_scope)
	return _discriminator(layer, depth-1, True, training)

def _losses(depth, img_g, img_h, disc_g, disc_r, disc_h):
	with tf.variable_scope("Loss%d"%depth):
		mean_g = tf.reduce_mean(disc_g)
		mean_r = tf.reduce_mean(disc_r)
		losses_g = -mean_g
		gradients = tf.gradients(disc_h, img_h)
		slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
		gradient_penalty = tf.reduce_mean(tf.square(slope-1.0))
		losses_d = tf.losses.compute_weighted_loss( \
			[mean_g, mean_r, gradient_penalty, tf.reduce_mean(tf.square(disc_g))], \
			[1.0, -1.0, 10.0, 0.001] \
		)
		return img_g, mean_r-mean_g, gradient_penalty, losses_g, losses_d

def _lerp_losses(lerp, loss1, loss2):
	return (loss2[0], *(_lerp(lerp, a, b) for a, b in zip(loss1[1:], loss2[1:])))

def _get_all_gans(seed, image, lerp, training=True):
	rnd = tf.random_uniform((BATCH_SIZE, 1, 1, 1), 0.0, 1.0)
	img_g = _generator(seed, 0, False, training)[1]
	img_r = tf.image.resize_bilinear(image, (16, 22))
	img_h = _lerp(rnd, img_r, img_g)
	disc_g = _discriminator(img_g, 0, False, training)
	disc_r = _discriminator(img_r, 0, True, training)
	disc_h = _discriminator(img_h, 0, True, training)
	old_loss = _losses(0, img_g, img_h, disc_g, disc_r, disc_h)
	yield old_loss
	for i in range(1, 4):
		img_next = _generator(seed, i, False, training)[1]
		size = (8*(2<<i), 11*(2<<i))
		img_g = _lerp(lerp, tf.image.resize_nearest_neighbor(img_g, size), img_next)
		img_r = tf.image.resize_bilinear(image, size)
		img_h = _lerp(rnd, img_r, img_g)
		disc_g = _discriminator(img_g, i, False, training)
		disc_r = _discriminator(img_r, i, True, training)
		disc_h = _discriminator(img_h, i, True, training)
		loss = _losses(i, img_g, img_h, disc_g, disc_r, disc_h)
		yield _lerp_losses(lerp, old_loss, loss)
		yield loss
		old_loss = loss

class PgGanGenerator(Generator):
	"""
        Neural network for generating art based on the PG-GAN architechture
	"""

	def __init__(self, iterations_per_lod=ITERATIONS_PER_LOD, learning_rate=LEARNING_RATE, training=True):
		super().__init__("pg-gan")
		with self.scope:
			self.seed = tf.random_uniform((BATCH_SIZE, 1, 1, 200))
			self.lerp = tf.to_float(tf.mod(self.global_step, iterations_per_lod)) / tf.to_float(iterations_per_lod)
			self.image = get_art_only_cropped(batch_size=BATCH_SIZE)
			self.gans = list(_get_all_gans(self.seed, self.image, self.lerp, training))
			with tf.variable_scope('Trainer'):
				self.trainer_g = []
				self.trainer_d = []
				adam_d = tf.train.AdamOptimizer(learning_rate, 0.0, 0.9, name='AdamD')
				adam_g = tf.train.AdamOptimizer(learning_rate*0.5, 0.0, 0.9, name='AdamG')
				for i, g in enumerate(self.gans):
					#Only use those variables and calculations that are needed
					scope_g = self.name+'/Generator[0-%d]'%i
					vars_g = tf.trainable_variables(scope_g)
					upds_g = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope_g)
					scope_d = self.name+'/Discriminator[0-%d]'%i
					vars_d = tf.trainable_variables(scope_d)
					upds_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope_d)
					#Add to the trainer lists
					with tf.control_dependencies(upds_d+upds_g):
						self.trainer_d.append(adam_d.minimize(g[-1], var_list=vars_d))
						self.trainer_g.append(adam_g.minimize(g[-2], global_step=self.global_step, var_list=vars_g))
			with tf.variable_scope('Summary'):
				#Summaries through variables because of different resolutions
				self.measure = tf.get_variable('Distance', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.gradient_penalty = tf.get_variable('GradientPenalty', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.loss_d = tf.get_variable('LossD', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.loss_g = tf.get_variable('LossG', [], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.generated = tf.get_variable('Generated', [4, IMAGE_HEIGHT, IMAGE_WIDTH, 3], tf.float32, trainable=False, initializer=tf.initializers.zeros)
				self.summaries = []
				self.measure_update = []
				for img, diff, gp, lg, ld in self.gans:
					measure_update = tf.assign(self.measure, 0.9*self.measure+0.1*diff)
					self.measure_update.append(measure_update)
					self.summaries.append(tf.group(
						measure_update,
						tf.assign(self.loss_d, ld),
						tf.assign(self.loss_g, lg),
						tf.assign(self.gradient_penalty, gp),
						tf.assign(self.generated, tf.image.resize_nearest_neighbor(img[:4,:,:,:], [IMAGE_HEIGHT, IMAGE_WIDTH]))
					))
			tf.summary.scalar("Distance", self.measure)
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
		lod = min(step // ITERATIONS_PER_LOD, 7)
		for _ in range(5):
			session.run(self.trainer_d[lod])
		if summary:
			_, _, step, res = session.run([self.trainer_g[lod], self.summaries[lod], self.global_step, self.measure])
			self.add_summary(session.run(self.summary), step)
		else:
			_, _, step, res = session.run([self.trainer_g[lod], self.measure_update[lod], self.global_step, self.measure])
		return step, res

