import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd

import edward as ed
from edward.models import Normal, Bernoulli, PointMass
from tensorflow.contrib.tensorboard.plugins import projector
from utils import *


class bayesian_emb_model():
    def __init__(self, d, K, sigma, sess, logdir):
        self.K = K
        self.sess = sess
        self.logdir = logdir
        self.sigma = sigma

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.target_placeholder = tf.placeholder(tf.int32)
                self.context_placeholder = tf.placeholder(tf.int32)
                self.labels_placeholder = tf.placeholder(tf.int32, shape=[d.n_minibatch])
                self.ones_placeholder = tf.placeholder(tf.int32)
                self.zeros_placeholder = tf.placeholder(tf.int32)
                self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])

            # Index Masks
            with tf.name_scope('priors'):
                self.U = Normal(loc=tf.zeros((d.L, self.K), dtype=tf.float32),
                                scale=tf.ones((d.L, self.K), dtype=tf.float32))
                self.V = Normal(loc=tf.zeros((d.L, self.K), dtype=tf.float32),
                                scale=tf.ones((d.L, self.K), dtype=tf.float32))

        with tf.name_scope('natural_param'):
            # Taget and Context Indices
            with tf.name_scope('target_word'):
                pos_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.ones(self.labels_placeholder.shape, dtype=tf.int32)))
                pos_words = tf.gather(self.target_placeholder, pos_indexes)
                self.p_rhos = tf.nn.embedding_lookup(self.U, pos_words)
                pos_contexts = tf.gather(self.context_placeholder, pos_indexes)
                self.pos_ctx_alpha = tf.nn.embedding_lookup(self.V, pos_contexts)

            with tf.name_scope('negative_samples'):
                neg_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.zeros(self.labels_placeholder.shape, dtype=tf.int32)))
                neg_words = tf.gather(self.target_placeholder, neg_indexes)
                self.n_rho = tf.nn.embedding_lookup(self.U, neg_words)
                neg_contexts = tf.gather(self.context_placeholder, neg_indexes)
                self.neg_ctx_alpha = tf.nn.embedding_lookup(self.V, neg_contexts)

            # Natural parameter
            self.p_eta = tf.reduce_sum(tf.multiply(self.p_rhos, self.pos_ctx_alpha), -1)
            self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, self.neg_ctx_alpha), -1)

        self.y_pos = Bernoulli(logits=self.p_eta)
        self.y_neg = Bernoulli(logits=self.n_eta)

        # INFERENCE
        sigma_init_array = np.full((d.L, 1), self.sigma, dtype=np.float32)
        self.sigU = tf.nn.softplus(
            tf.matmul(tf.get_variable("sigU", initializer=sigma_init_array), tf.ones([1, self.K])),
            name="sigmasU")
        self.sigV = tf.nn.softplus(
            tf.matmul(tf.get_variable("sigV", initializer=sigma_init_array), tf.ones([1, self.K])),
            name="sigmasV")
        #self.locV = tf.get_variable("qV/loc", [d.L, self.K], initializer=tf.zeros_initializer())

        self.qU = Normal(loc=d.embedding_matrix, scale=self.sigU)
        self.qV = Normal(loc=d.embedding_matrix, scale=self.sigV)

        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV},
                                 data={self.y_pos: self.ones_placeholder,
                                       self.y_neg: self.zeros_placeholder
                                       })
        with self.sess.as_default():
            tf.global_variables_initializer().run()
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()

        alpha = config.embeddings.add()
        alpha.tensor_name = 'qU/loc'
        alpha.metadata_path = '../vocab.tsv'
        rho = config.embeddings.add()
        rho.tensor_name = 'qV/loc'
        rho.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    def dump(self, fname, data):
        with self.sess.as_default():
            dat = {'rhos': data.embedding_matrix,
                   'alpha': self.qV.loc.eval(),
                   'sigma_rhos': self.sigU.eval()[:, 0],
                   'sigma_alphas': self.sigV.eval()[:, 0],
                   'words': self.build_words_list(data.words, len(self.sigU.eval())),
                   'counts': data.counter}
            pickle.dump(dat, open(fname, "ab+"))

    def build_words_list(self, labels, list_length):
        if len(labels) < list_length:
            empty_list = [''] * (list_length - len(labels))
            labels.extend(empty_list)
        return labels

class bayesian_emb_inference_model():
    def __init__(self, d, sess, logdir, sigmas = None):
        self.K = d.K
        self.sess = sess
        self.logdir = logdir

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.target_placeholder = tf.placeholder(tf.int32)
                self.context_placeholder = tf.placeholder(tf.int32)
                self.labels_placeholder = tf.placeholder(tf.int32)
                self.batch_size = tf.placeholder(tf.int32)
                self.pos_empiric_probs = tf.placeholder(tf.float32)
                self.neg_empiric_probs = tf.placeholder(tf.float32)
                self.pos_ctxt_probs = tf.placeholder(tf.float32)
                self.neg_ctxt_probs = tf.placeholder(tf.float32)

            # Index Masks
            with tf.name_scope('priors'):
                if sigmas is None:
                    sigmas = tf.zeros((d.L, self.K), dtype=tf.float32)
                else:
                    sigmas = tf.tile(tf.expand_dims(sigmas, 1), [1, self.K])

                self.U = Normal(loc=d.embedding_matrix,
                                scale=sigmas)
                self.V = Normal(loc=d.embedding_matrix,
                                scale=sigmas)

        with tf.name_scope('natural_param'):
            # Taget and Context Indices
            with tf.name_scope('target_word'):
                pos_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.ones(self.batch_size, dtype=tf.int32)))
                pos_words = tf.gather(self.target_placeholder, pos_indexes)
                self.p_rhos = tf.nn.embedding_lookup(self.U, pos_words)
                pos_contexts = tf.gather(self.context_placeholder, pos_indexes)
                self.pos_ctx_alpha = tf.nn.embedding_lookup(self.V, pos_contexts)

            with tf.name_scope('negative_samples'):
                neg_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.zeros(self.batch_size, dtype=tf.int32)))
                neg_words = tf.gather(self.target_placeholder, neg_indexes)
                self.n_rho = tf.nn.embedding_lookup(self.U, neg_words)
                neg_contexts = tf.gather(self.context_placeholder, neg_indexes)
                self.neg_ctx_alpha = tf.nn.embedding_lookup(self.V, neg_contexts)

            # Natural parameter
            self.p_eta = tf.reduce_sum(tf.multiply(self.p_rhos, self.pos_ctx_alpha), -1)
            self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, self.neg_ctx_alpha), -1)

        self.y_pos = Bernoulli(logits=self.p_eta)
        self.y_neg = Bernoulli(logits=self.n_eta)

        self.prob_pos = tf.squeeze(self.y_pos.prob(1.0))
        self.prob_neg = tf.squeeze(self.y_neg.prob(0.0))
        self.mul_prob_pos = tf.multiply(self.prob_pos, self.pos_ctxt_probs)
        self.mul_prob_neg = tf.multiply(self.prob_neg, self.neg_ctxt_probs)

        self.log_prob_pos = tf.log(tf.reduce_sum(self.mul_prob_pos))
        self.log_prob_neg = tf.log(tf.reduce_sum(self.mul_prob_neg))

        self.pos_empiric = tf.reduce_sum(tf.multiply(self.pos_empiric_probs, self.pos_ctxt_probs))
        self.neg_empiric = tf.reduce_sum(tf.multiply(self.neg_empiric_probs, self.neg_ctxt_probs))

        self.entropy_pos = tf.negative(tf.reduce_sum(tf.multiply(self.pos_empiric, self.log_prob_pos)))
        self.entropy_neg = tf.negative(tf.reduce_sum(tf.multiply(self.neg_empiric, self.log_prob_neg)))

        self.perplexity_pos = tf.exp(self.entropy_pos)
        self.perplexity_neg = tf.exp(self.entropy_neg)

        with self.sess.as_default():
            tf.global_variables_initializer().run()