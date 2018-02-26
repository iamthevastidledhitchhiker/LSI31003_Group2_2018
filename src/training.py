import math
import numpy as np

import keras.backend as K
import tensorflow as tf

#	TODO: write this function
#	using an autoencoder to remove noise
def denoise(source_data, target_data, max_zeros = 1):
	def remove_zeros(data):
		to_keep = np.sum((data == 0), axis = 1) <= max_zeros
		return data[to_keep]

	less_zeros = map(remove_zeros, [source_data, target_data])
	ae_target = np.concatenate(less_zeros, axis = 0)
#	TODO

#	TODO: add necessary parameters passed to denoise
def load_data(source_path, target_path, use_denoise = False):
	def load_and_log(path):
		from numpy import genfromtxt as load_csv
		from Calibration_Util.DataHandler import preProcessCytofData as pp

		return pp(load_csv(path, delimiter = ','))

	def preprocess(source, target):
		from sklearn.preprocessing import StandardScaler

		preprocessor = StandardScaler().fit(source)
		return [ preprocessor.transform(data) for data in [source, target] ]

	source, target = map(load_and_log, [source_path, target_path])
	if use_denoise:
	#	TODO: call denoise
		pass

	return preprocess(source, target)

def create_network(input_dim, layer_sizes = [25, 25, 25], l2_penalty = 1e-2):
	from keras.layers import Input
	from keras.models import Model

#	I heard that 'blockchain' and 'machine learning' are the current buzz-words,
#	so here is a blockchain in a machine learning context.
	def block_chain(block_in, layer_size):
		from keras.layers import add

		def add_layer(block_in, size):
			from keras.layers import Dense, Activation

			from keras.layers.normalization import BatchNormalization
			from keras.initializers import RandomNormal
			from keras.regularizers import l2

			normalizer = BatchNormalization()(block_in)
			activator = Activation('relu')(normalizer)
			return Dense(size,
				activation = 'linear',
				kernel_regularizer = l2(l2_penalty),
				kernel_initializer = RandomNormal(stddev = 1e-4)
				)(activator)

	#	'input_dim' is already part of this namespace
		layer1 = add_layer(block_in, layer_size)
		layer2 = add_layer(layer1, input_dim)

		return add([layer2, block_in])

	input_block = output_block = Input(shape = (input_dim, ))
	for size in layer_sizes:
		output_block = block_chain(output_block, size)

	mmd_net = Model(inputs = input_block, outputs = output_block)
	return (input_block, output_block), mmd_net

def train_network(mmd_net, source, target, last_block, verbose = False):
	from keras.callbacks import LearningRateScheduler, EarlyStopping
	from keras.optimizers import rmsprop

	from CostFunctions import MMD
	from Monitoring import monitorMMD

	def step_decay(epoch, initial_lrate = 0.001, drop = 0.1, epochs_drop = 150.0):
		return initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

	def mmd(y_true, y_pred):
	#	'last_block' and 'target' are already part of this namespace
		return MMD(last_block, target, MMDTargetValidation_split = 0.1).KerasCost(y_true, y_pred)

	learning_rate = LearningRateScheduler(step_decay)
	mmd_net.compile(optimizer = rmsprop(lr = 0.0), loss = mmd)
	K.get_session().run(tf.global_variables_initializer())

	labels = np.zeros(source.shape[0])
	callbacks = [learning_rate]
	callbacks.append(monitorMMD(source, target, mmd_net.predict))
	callbacks.append(EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto'))

	mmd_net.fit(source,
		labels,
		epochs = 500,
		batch_size = 512,
		validation_split = 0.1,
		verbose = verbose,
		callbacks = callbacks)
