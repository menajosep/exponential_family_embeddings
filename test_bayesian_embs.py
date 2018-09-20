import os
import unittest

from tensorflow.python.training.adam import AdamOptimizer
import tensorflow as tf

from models import bayesian_emb_model
from test_utils import bayessian_bern_emb_data_deterministic, bayessian_bern_emb_data_deterministic_inverted
from utils import get_logger
import edward as ed
import shutil


class DeterministicSamplingTestCase(unittest.TestCase):
    """Tests for `primes.py`."""

    def setUp(self):
        self.sess = ed.get_session()
        self.logger = get_logger()
        self.dir_name = "DeterministicSamplingTestCase"
        self.sigma = 8
        self.epochs = 100
        self.context_size = 10
        self.negative_samples = 10
        self.dimension = 300
        self.minibatch = 256
        self.repetitions = 10000
        # DATA
        self.det_data = bayessian_bern_emb_data_deterministic(self.logger,
                                                              self.context_size,
                                                              self.negative_samples,
                                                              self.dimension,
                                                              self.minibatch,
                                                              self.repetitions)
        # MODEL
        self.logger.debug('....build model')
        self.model = bayesian_emb_model(self.det_data, self.dimension, self.sigma, self.sess, self.dir_name)

    def tearDown(self):
        del self.det_data

    def test_no_uncertainty(self):
        sigmas = self.training()
        self.assertTrue(sigmas[0] == 1.0,
                        msg='{} should have uncertainty equals to 1'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[1] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[2] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))

    def test_no_uncertainty_noise(self):
        sigmas = self.training(9)
        self.assertTrue(sigmas[0] == 1.0,
                        msg='{} should have uncertainty equals to 1'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[1] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[2] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))

    def test_inverted(self):
        # DATA
        self.det_data = bayessian_bern_emb_data_deterministic_inverted(self.logger,
                                                              self.context_size,
                                                              self.negative_samples,
                                                              self.dimension,
                                                              self.minibatch,
                                                              self.repetitions)
        # MODEL
        self.logger.debug('....build model')
        self.model = bayesian_emb_model(self.det_data, self.dimension, self.sigma, self.sess, self.dir_name)
        sigmas = self.training()
        self.assertTrue(sigmas[0] == 1.0,
                        msg='{} should have uncertainty equals to 1'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[1] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))
        self.assertTrue(sigmas[2] < 0.005,
                        msg='{} should be have low uncertainty'.format(self.det_data.reverse_dictionary[1]))

    def get_n_iters(self, ns, n_epochs, batch_size, dictionary):
        n_samples = 2 * (len(dictionary)-1) * ns * self.repetitions
        n_batches = n_samples / batch_size
        if len(dictionary) % batch_size > 0:
            n_batches += 1
        return int(n_batches) * n_epochs, int(n_batches)

    def training(self, noise=0):
        # TRAINING
        n_iters, n_batches = self.get_n_iters(self.negative_samples, self.epochs, self.minibatch,
                                         self.det_data.dictionary)
        # kl_scaling_weights = get_kl_weights(n_batches)
        self.logger.debug('init training number of iters ' + str(n_iters) + ' and batches ' + str(n_batches))
        self.model.inference.initialize(n_samples=1, n_iter=n_iters, logdir=self.model.logdir,
                                        scale={self.model.y_pos: n_batches,
                                               self.model.y_neg: n_batches / self.negative_samples},
                                        optimizer=AdamOptimizer(learning_rate=self.model.learning_rate_placeholder)
                                        )
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.logger.debug('....starting training')
        sigmas = []
        for epoch in range(self.epochs):
            self.det_data.batch = self.det_data.batch_generator(self.minibatch, noise)
            for batch in range(n_batches):
                info_dict = self.model.inference.update(feed_dict=self.det_data.feed(self.model.target_placeholder,
                                                                                     self.model.context_placeholder,
                                                                                     self.model.labels_placeholder,
                                                                                     self.model.ones_placeholder,
                                                                                     self.model.zeros_placeholder,
                                                                                     self.model.learning_rate_placeholder,
                                                                                     self.minibatch,
                                                                                     0.01))
                self.model.inference.print_progress(info_dict)
            sigmas = self.model.sigU.eval()[:, 0]
        self.logger.debug('Done')
        return sigmas


if __name__ == '__main__':
    unittest.main()