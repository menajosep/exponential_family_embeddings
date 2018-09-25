from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import tensorflow as tf

from data import *
from mle_models import *
from args import *

logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = tf.Session()

# DATA
d = bern_emb_data(args.in_file, args.cs, args.ns, args.mb, args.L, args.emb_type, args.word2vec_file, args.glove_file,
                           args.fasttext_file, logger)

# MODEL
m = bern_emb_model(d, args.K, args.sig, sess, dir_name)


# TRAINING
train_loss = np.zeros(args.n_iter)

for i in range(args.n_iter):
    for ii in range(args.n_epochs):
        sess.run([m.train], feed_dict=d.feed(m.placeholders))
    summary, _ , train_loss[i] = sess.run([m.summaries, m.train, m.loss], feed_dict=d.feed(m.placeholders))
    m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
    m.train_writer.add_summary(summary, i)
    print("iteration {:d}/{:d}, train loss: {:0.3f}\n".format(i, args.n_iter, train_loss[i])) 

print('training finished. Results are saved in '+dir_name)
m.dump(dir_name+"/variational.dat")
m.plot_params(dir_name, d.labels[:500])
logger.debug('Done')
