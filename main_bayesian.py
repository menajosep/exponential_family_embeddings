from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.adam import AdamOptimizer
import matplotlib
matplotlib.use('Agg')

from args import *
from data import *
from models import *


def get_n_iters():
    n_batches = len(d.word_target) / d.n_minibatch
    if len(d.word_target) % d.n_minibatch > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


def get_kl_weights(n_batches):
    weights = np.full(n_batches, 1./((2 ** 1000) - 1), dtype=np.float64)
    weights_lim = min(n_batches, 1000)
    for i in range(weights_lim):
        weight = (2 ** (weights_lim - i)) / ((2 ** n_batches) - 1)
        weights[i] = weight
    return weights


args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = bayessian_bern_emb_data(args.in_file, args.cs, args.ns, args.mb, args.L, args.K,
                            args.emb_type, args.word2vec_file, args.glove_file,
                            args.fasttext_file, args.custom_file, dir_name)

# MODEL
m = bayesian_emb_model(d, d.K, sess, dir_name)
sigmas_list = list()


# TRAINING
n_iters, n_batches = get_n_iters()
kl_scaling_weights = get_kl_weights(n_batches)

m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       kl_scaling={m.y_pos: kl_scaling_weights, m.y_neg: kl_scaling_weights},
                       optimizer=AdamOptimizer(learning_rate=0.01)
                       )
init = tf.global_variables_initializer()
sess.run(init)

for i in range(m.inference.n_iter):
    info_dict = m.inference.update(feed_dict=d.feed(m.target_placeholder,
                                                    m.context_placeholder,
                                                    m.labels_placeholder,
                                                    m.ones_placeholder,
                                                    m.zeros_placeholder,
                                                    True))
    m.inference.print_progress(info_dict)
    if i % 10000 == 0:
        m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
        sigmas = m.sigU.eval()[:, 0]
        sigmas_list.append(sigmas)
        pickle.dump(sigmas_list, open(dir_name + "/sigmas.dat", "wb+"))
        if is_good_embedding(sigmas):
            break

m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
print('training finished. Results are saved in ' + dir_name)
m.dump(dir_name + "/variational.dat", d)

print('Done')