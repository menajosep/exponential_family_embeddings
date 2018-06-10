from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.adam import AdamOptimizer

from args import *
from data import *
from models import *


def get_n_iters():
    n_batches = len(d.word_target) / d.n_minibatch
    if len(d.word_target) % d.n_minibatch > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


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

m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       kl_scaling={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
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
        pickle.dump(sigmas_list, open(dir_name + "/sigmas.dat", "w+"))
        if is_goog_embedding(sigmas):
            break

m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
print('training finished. Results are saved in ' + dir_name)
m.dump(dir_name + "/variational.dat", d)

print('Done')