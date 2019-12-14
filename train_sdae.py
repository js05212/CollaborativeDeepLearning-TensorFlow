import numpy as np
import tensorflow as tf
import scipy.io
import logging
from lib.sdae import SDAE
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("sdae.log")

logging.info('loading data')
variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
data = variables['X']
idx = np.random.rand(data.shape[0]) < 0.8 # can change to 1.0
train_X = data[idx]
test_X = data[~idx]
logging.info('initializing sdae model')
# epoch=[50, 50]
	#activations=['sigmoid'], epoch=[50], epoch_joint=100, 
model = SDAE(input_dim=8000, dims=[50], z_dim=50, 
	activations=['sigmoid'], epoch=[50], epoch_joint=100, 
	noise='mask-0.3', dropout=0.8, loss='cross-entropy', lr=0.01, batch_size=128, print_step=1)
logging.info('fitting data starts...')
model.fit(train_X, test_X)
# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)
