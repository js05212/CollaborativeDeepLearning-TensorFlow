import numpy as np
import lib.utils as utils
import tensorflow as tf
import logging

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']
def xavier_init(fan_in, fan_out, dtype=tf.float32, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class SDAE:
    """A stacked denoising autoencoder"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, input_dim, dims, z_dim, activations, epoch=1000, epoch_joint=1000, noise=None, dropout=0.0,
                loss='cross-entropy',
                lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.dropout = dropout
        self.epoch = epoch
        self.epoch_joint = epoch_joint
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.n_z = z_dim
        self.input_dim = input_dim
        self.weights, self.biases = [], []
        self.de_weights, self.de_biases = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(
                    frac * len(i))), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, data_x, x_valid=None):
        x = data_x
        for i in range(self.depth):
            logging.info('Layer {0}'.format(i + 1))
            x = self.run(data_x=x,
                         activation=self.activations[i],
                         hidden_dim=self.dims[i],
                         epoch=self.epoch[
                             i], loss=self.loss,
                         batch_size=self.batch_size,
                         lr=self.lr, print_step=self.print_step)
        # fit latent layer
        self.run_latent(data_x=x, hidden_dim=self.n_z, batch_size=self.batch_size,
            lr=self.lr, epoch=50, print_step=self.print_step)
        self.run_all(data_x=data_x, lr=self.lr, batch_size=self.batch_size, 
            epoch=self.epoch_joint, print_step=self.print_step, x_valid=x_valid)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        # x = tf.constant(data, dtype=tf.float32)
        input_dim = len(data[0])
        data_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x = data_x
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return sess.run(x, feed_dict={data_x: data})
        # return x.eval(session=sess)


    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    # train SDAE
    def run_all(self, data_x, lr, batch_size, epoch, print_step=100, x_valid=None):
        tf.reset_default_graph()
        n = data_x.shape[0]
        input_dim = len(data_x[0])
        num_iter = int(n / batch_size)
        with tf.variable_scope("inference"):
            rec = {'W1': tf.get_variable("W1", initializer=tf.constant(self.weights[0]), dtype=tf.float32),
                'b1': tf.get_variable("b1", initializer=tf.constant(self.biases[0]), dtype=tf.float32),
                'W_z_mean': tf.get_variable("W_z_mean", initializer=tf.constant(self.weights[1]), dtype=tf.float32),
                'b_z_mean': tf.get_variable("b_z_mean", initializer=tf.constant(self.biases[1]), dtype=tf.float32)}

        with tf.variable_scope("generation"):
            gen = {'W1': tf.get_variable("W1", initializer=tf.constant(self.de_weights[1]), dtype=tf.float32),
                'b1': tf.get_variable("b1", initializer=tf.constant(self.de_biases[1]), dtype=tf.float32),
                'W_x': tf.transpose(rec['W1']),
                'b_x': tf.get_variable("b_x", [input_dim], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        weights = []
        weights += [rec['W1'], rec['b1'], rec['W_z_mean'],
                        rec['b_z_mean']]#, rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        weights += [gen['W1'], gen['b1'], gen['b_x']]
        saver = tf.train.Saver(weights)

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_') # add in 2019
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')
        h1 = self.activate(
            tf.matmul(x, rec['W1']) + rec['b1'], self.activations[0])
        dropout_h1 = tf.nn.dropout(h1, dropout_prob)
        z_mean = tf.matmul(dropout_h1, rec['W_z_mean']) + rec['b_z_mean']
        z = z_mean
        dropout_z = tf.nn.dropout(z, dropout_prob)
        h1 = self.activate(
            tf.matmul(dropout_z, gen['W1']) + gen['b1'], self.activations[0])
        dropout_h1 = tf.nn.dropout(h1, dropout_prob)
        x_recon = tf.matmul(dropout_h1, gen['W_x']) + gen['b_x']
        x_recon = tf.nn.sigmoid(x_recon, name='x_recon')
        gen_loss = -tf.reduce_mean(tf.reduce_sum(x_ * tf.log(tf.maximum(x_recon, 1e-10)) 
            + (1-x_) * tf.log(tf.maximum(1 - x_recon, 1e-10)),1))
        latent_loss = 0.0
        loss = gen_loss
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_x_, ids = utils.get_batch(data_x, batch_size)
                b_x = self.add_noise(b_x_) # denoising
                _, l, gl = sess.run((train_op, loss, gen_loss), feed_dict={x: b_x, x_: b_x_, dropout_prob: self.dropout})
            if (i + 1) % print_step == 0:
                if x_valid is None:
                    logging.info('epoch {0}: batch loss = {1}, gen_loss={2}'.format(i, l, gl))
                else:
                    valid_loss = self.validation(x_valid, sess, gen_loss, x, x_, dropout_prob, batch_size)
                    logging.info('epoch {0}: batch loss = {1}, gen_loss={2}, valid_loss={3}'.format(i, l, gl, valid_loss))

        weight_path = "model/pretrain_cdl"
        saver.save(sess, weight_path)
        logging.info("Weights saved at " + weight_path)

    def validation(self, data_x, sess, gen_loss, x, x_, dropout_prob, batch_size):
        n_samples = data_x.shape[0]
        num_batches = int(1.0*n_samples/self.batch_size)
        n_samples = num_batches * batch_size
        valid_loss = 0.
        for i in range(num_batches):
            ids = range(i*batch_size, (i+1)*batch_size)
            x_b = data_x[ids]
            gl = sess.run(gen_loss, feed_dict={x: x_b, x_: x_b, dropout_prob: 1.0})
            valid_loss += gl / n_samples * batch_size
        return valid_loss

    # train the deepest AE
    def run_latent(self, data_x, hidden_dim, batch_size, lr, epoch, print_step=100):
        tf.reset_default_graph()
        n = data_x.shape[0]
        input_dim = len(data_x[0])
        num_iter = int(n / batch_size)
        sess = tf.Session()
        rec = { 'W_z_mean': tf.get_variable("W_z_mean", [self.dims[0], self.n_z], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean': tf.get_variable("b_z_mean", [self.n_z], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        gen = {'W2': tf.get_variable("W2", [self.n_z, self.dims[0]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2", [self.dims[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')
        z_mean = tf.matmul(x, rec['W_z_mean']) + rec['b_z_mean']
        z = z_mean
        dropout_z = tf.nn.dropout(z, dropout_prob)
        x_recon = tf.matmul(dropout_z, gen['W2']) + gen['b2']
        x_recon = tf.nn.sigmoid(x_recon, name='x_recon')
        gen_loss = -tf.reduce_mean(tf.reduce_sum(x_ * tf.log(tf.maximum(x_recon, 1e-10)) 
            + (1-x_) * tf.log(tf.maximum(1 - x_recon, 1e-10)),1))
        loss = gen_loss
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_x_, ids = utils.get_batch(data_x, batch_size)
                b_x = self.add_noise(b_x_)
                _, l, gl = sess.run((train_op, loss, gen_loss), feed_dict={x: b_x, x_: b_x_, dropout_prob: self.dropout})
            if (i + 1) % print_step == 0:
                logging.info('epoch {0}: batch loss = {1}, gen_loss={2}'.format(i, l, gl))

        self.weights.append(sess.run(rec['W_z_mean']))
        self.biases.append(sess.run(rec['b_z_mean']))
        self.de_weights.append(sess.run(gen['W2']))
        self.de_biases.append(sess.run(gen['b2']))

    def run(self, data_x, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        n = data_x.shape[0]
        num_iter = int(n / batch_size)
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')
        encode = {'weights': tf.Variable(xavier_init(input_dim, hidden_dim, dtype=tf.float32)),
            'biases': tf.Variable(tf.zeros([hidden_dim],
                                                      dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.zeros([input_dim],dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}
        encoded = self.activate(
            tf.matmul(x, encode['weights']) + encode['biases'], activation)
        dropout_enc = tf.nn.dropout(encoded, dropout_prob)
        decoded = tf.matmul(dropout_enc, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_ - decoded), 1))
        elif loss == 'cross-entropy':
            decoded = tf.nn.sigmoid(decoded, name='decoded')
            loss = -tf.reduce_mean(tf.reduce_sum(x_ * tf.log(tf.maximum(decoded, 1e-16)) + (1-x_)*tf.log(tf.maximum(1-decoded, 1e-16)), 1))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for it in range(num_iter):
                b_x_, ids = utils.get_batch(data_x, batch_size)
                b_x = self.add_noise(b_x_)
                _, l = sess.run((train_op, loss), feed_dict={x: b_x, x_: b_x_, dropout_prob: self.dropout})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: b_x_, x_: b_x_, dropout_prob: 1.0})
                logging.info('epoch {0}: batch loss = {1}'.format(i, l))

        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        self.de_weights.append(sess.run(decode['weights']))
        self.de_biases.append(sess.run(decode['biases']))

        return sess.run(encoded, feed_dict={x: data_x, dropout_prob: 1.0})

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')
