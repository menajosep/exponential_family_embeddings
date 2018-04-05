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

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = bern_emb_data(args.cs, args.ns, args.mb, args.L)

# MODEL
m = bayesian_emb_model(d, args.K, args.sig, sess, dir_name)


def get_n_iters():
    n_batches = len(d.data) / d.n_minibatch
    if len(d.data) % d.n_minibatch > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


# TRAINING
n_iters, n_batches = get_n_iters()
m.inference.initialize(n_samples=5, n_iter=n_iters, logdir=m.logdir)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(m.inference.n_iter):
    info_dict = m.inference.update(feed_dict=d.feed_with_labels(m.placeholders, m.y_pos_ph, m.y_neg_ph))
    m.inference.print_progress(info_dict)
    if i % n_batches == 0:
        m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
print('training finished. Results are saved in '+dir_name)
m.dump(dir_name+"/variational.dat", d.labels, 100)
