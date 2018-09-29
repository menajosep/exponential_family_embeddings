from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from data import *
from mle_models import *
from args import *

logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = bern_emb_data(args.in_file, args.cs, args.ns, args.mb, args.L, args.emb_type, args.word2vec_file, args.glove_file,
                           args.fasttext_file, logger)
pickle.dump(d, open(dir_name + "/data.dat", "wb+"))
logger.debug('....load dataset')
d = pickle.load(open(dir_name + "/data.dat", "rb+"))
# d = pickle.load(open("data/recipes/data.dat", "rb+"))
logger.debug('....load embeddings matrix')
d.load_embeddings(args.emb_type, args.word2vec_file, args.glove_file,
                           args.fasttext_file, logger)
d.batch = d.batch_generator()

# MODEL
m = bern_emb_model_vi(d, args.sig, sess, dir_name)


# TRAINING
n_batches = int(len(d.data)/args.mb)
if len(d.data) % args.mb > 0:
    n_batches += 1
n_iters = args.n_epochs * n_batches

m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir, optimizer=AdamOptimizer(learning_rate=args.lr))
init = tf.global_variables_initializer()
sess.run(init)
n_iters = int(len(d.data)/args.mb)
for i in range(args.n_epochs):
    for ii in range(n_batches):
        info_dict = m.inference.update(feed_dict=d.feed(m.placeholders, m.ones_placeholder, m.zeros_placeholder))
        m.inference.print_progress(info_dict)
    m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)

print('training finished. Results are saved in '+dir_name)
sigmas = m.sigrho.eval()[:, 0]
pickle.dump(sigmas, open(dir_name + "/sigmas.dat", "wb+"))
m.dump(dir_name+"/variational.dat")
logger.debug('Done')
