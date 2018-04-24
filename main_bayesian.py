from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import tensorflow as tf

from data import *
from models import *
from args import *
from collections import Counter
import pandas as pd
import math
import operator

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = bayessian_bern_emb_data(args.cs, args.ns, args.mb, args.L, args.in_file)

# MODEL
m = bayesian_emb_model(d, args.K, sess, dir_name, args.emb_file)


def get_n_iters():
    n_batches = len(d.chars_target_train) / d.n_minibatch
    if len(d.chars_target_train) % d.n_minibatch > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


# TRAINING
n_iters, n_batches = get_n_iters()

m.inference.initialize(n_samples=10, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches/args.ns},
                       kl_scaling={m.y_pos: n_batches, m.y_neg: n_batches/args.ns}
                      )
init = tf.global_variables_initializer()
sess.run(init)
for i in range(m.inference.n_iter):
    info_dict = m.inference.update(feed_dict=d.feed(m.target_placeholder,
                                                    m.context_placeholder,
                                                    m.labels_placeholder,
                                                    m.ones_placeholder,
                                                    m.zeros_placeholder
                                                   ))
    m.inference.print_progress(info_dict)
    if i % n_batches == 0:
        m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
print('training finished. Results are saved in '+dir_name)
m.dump(dir_name +"/variational.dat", d.chars, 100)

# Build samples from inferred posterior.
n_samples = 500
probs = []
char_target = np.array([d.dictionary[' ']])
char_context = np.array([d.dictionary['j']])
for _ in range(n_samples):
    target_embs = m.qU.sample()
    rho = tf.nn.embedding_lookup(target_embs, char_target)
    rho = (rho / tf.norm(rho))
    context_embs = m.qV.sample()
    alpha = tf.nn.embedding_lookup(context_embs, char_context)
    alpha = (alpha / tf.norm(alpha))
    prob = tf.reduce_sum(tf.multiply(rho, alpha), -1)
    probs.append(tf.sigmoid(prob))

outputs = tf.stack(probs).eval()
outputs.mean()
# outputs

# sort by the sigmas
sigmas = list(m.sigV.eval()[:,0])
sorted_x = sorted(d.dictionary.items(), key=operator.itemgetter(1))
dictionary_keys, _ = zip(*sorted_x)
sigma_df = pd.DataFrame(
    {'letter': dictionary_keys,
     'sigma': sigmas
    })
list(sigma_df.letter)
letters_by_sigma = sigma_df.sort_values(by='sigma')

#sort by entropy
filename = '/Users/jose.mena/dev/personal/data/wiki/wiki106.txt.zip'
chars = read_data_as_chars(filename)
count_chars = Counter(chars)
count_pairs = Counter(zip(chars, chars[1:]))
total_pairs = np.array(count_pairs.values()).sum()

def weighted_information(value):
    prob = 1. * value / total_pairs
    return -((prob) * (math.log(prob)))

values = count_pairs.values()
first_letters, second_letters = zip(*count_pairs.keys())
pairs_df = pd.DataFrame(
    {'first_letter': first_letters,
     'second_letter': second_letters,
     'occurs': values
    })
pairs_df['entropy'] = pairs_df['occurs'].apply(weighted_information)
entropy = pairs_df.groupby('first_letter').sum()
letters_by_entropy = entropy.sort_values(by='entropy', ascending=False)
letters_by_entropy.index.name = None

print('Done')