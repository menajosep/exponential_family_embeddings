from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import matplotlib
matplotlib.use('Agg')

from args import *
from data import *
from models import *


def get_n_iters():
    n_batches = (len(d.dictionary)*(int(args.ns) +1)) / d.n_minibatch
    if len(d.dictionary) % d.n_minibatch > 0:
        n_batches += 1
    return int(n_batches) * args.n_epochs, int(n_batches)


def get_kl_weights(n_batches):
    weights = np.full(n_batches, 1./((2 ** 1000) - 1), dtype=np.float64)
    weights_lim = min(n_batches, 1000)
    for i in range(weights_lim):
        weight = (2 ** (weights_lim - i)) / ((2 ** n_batches) - 1)
        weights[i] = weight
    return weights


def clr(clr_iteration, step_size, base_lr, max_lr):
    # Triangular2
    scale_fn = lambda x: 1 / (2. ** (x - 1))
    cycle = np.floor(1 + clr_iteration / (2 * step_size))
    x = np.abs(clr_iteration / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(cycle)


logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
# d = bayessian_bern_emb_data(args.in_file, args.cs, args.ns, args.mb, args.L, args.K,
#                             args.emb_type, args.word2vec_file, args.glove_file,
#                             args.fasttext_file, args.custom_file, dir_name, logger)
# pickle.dump(d, open(dir_name + "/data.dat", "wb+"))

# MODEL
# d = pickle.load(open(dir_name + "/data.dat", "rb+"))
d = pickle.load(open("fits/local/data.dat", "rb+"))
m = bayesian_emb_model(d, d.K, sess, dir_name)
sigmas_list = list()


# TRAINING
n_iters, n_batches = get_n_iters()
#kl_scaling_weights = get_kl_weights(n_batches)
learning_rates = []
for i in range(n_iters):
    learning_rates.append(clr(i, n_iters, 1e-4, 2))
#learning_rates = np.array(learning_rates)
logger.debug('init training number of iters '+str(n_iters)+' and batches '+str(n_batches))
m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       #kl_scaling={m.y_pos: kl_scaling_weights, m.y_neg: kl_scaling_weights},
                       optimizer=AdamOptimizer(learning_rate=m.learning_rate_placeholder)
                       )
init = tf.global_variables_initializer()
sess.run(init)
logger.debug('....starting training')
iteration = 0
distances = np.zeros([n_iters])
for i in range(args.n_epochs):
    d.batch = d.batch_generator(args.mb)
    for i in range(n_batches):
        info_dict = m.inference.update(feed_dict=d.feed(m.target_placeholder,
                                                        m.context_placeholder,
                                                        m.labels_placeholder,
                                                        m.ones_placeholder,
                                                        m.zeros_placeholder,
                                                        m.learning_rate_placeholder,
                                                        args.mb,
                                                        learning_rates[iteration]))
        iteration += 1
        m.inference.print_progress(info_dict)
        if iteration % 5 == 0:
            sigmas = m.sigU.eval()[:, 0]
            distance = get_distance(sigmas)
            distances[iteration] = distance
            if iteration % 10000 == 0:
                m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
                sigmas_list.append(sigmas)
                pickle.dump(sigmas_list, open(dir_name + "/sigmas.dat", "wb+"))
                if is_good_embedding(sigmas):
                    break

m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), i)
logger.debug('training finished. Results are saved in ' + dir_name)
m.dump(dir_name + "/variational.dat", d)
pickle.dump(distances, open(dir_name + "/distances.dat", "wb+"))
pickle.dump(learning_rates, open(dir_name + "/learning_rates.dat", "wb+"))

logger.debug('Done')
