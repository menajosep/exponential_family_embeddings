from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.adam import AdamOptimizer
import matplotlib
matplotlib.use('Agg')

from args import *
from data import *
from bayesian_models import *

FAKE_WORD = 'grijander'

logger = get_logger()

args, dir_name = parse_args_bayesian()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = bayessian_bern_emb_data(args.in_file, args.cs, args.ns, args.mb, args.L, args.K,
                           args.emb_type, args.word2vec_file, args.glove_file,
                           args.fasttext_file, args.custom_file,
                           args.fake_sentences, dir_name, logger)
logger.debug('....dump dataset')
pickle.dump(d, open(dir_name + "/data.dat", "wb+"))
logger.debug('....load dataset')
d = pickle.load(open(dir_name + "/data.dat", "rb+"))
# d = pickle.load(open("data/recipes/data.dat", "rb+"))
logger.debug('....load embeddings matrix')
if args.fake_word is not None:
    d.dictionary[FAKE_WORD] = len(d.dictionary)
    d.L = d.L + 1
    d.positive_word_sampling_indexes[d.dictionary[FAKE_WORD]] = d.positive_word_sampling_indexes[d.dictionary[args.fake_word]]
    d.negative_word_sampling_indexes[d.dictionary[FAKE_WORD]] = d.negative_word_sampling_indexes[d.dictionary[args.fake_word]]
d.load_embeddings(args.emb_type, args.word2vec_file, args.glove_file,
                           args.fasttext_file, args.custom_file, logger)
if args.fake_word is not None:
    d.embedding_matrix[d.dictionary[FAKE_WORD]] = d.embedding_matrix[d.dictionary[args.fake_word]]
# MODEL
logger.debug('....build model')
m = bayesian_emb_model(d, d.K, args.sigma, sess, dir_name)
sigmas_list = list()


# TRAINING
n_iters, n_batches = get_n_iters(args.ns, args.n_epochs, args.mb, d.dictionary)
#kl_scaling_weights = get_kl_weights(n_batches)
learning_rates = get_learning_rates(args.clr_type, n_iters, args.clr_cycles, args.base_lr, args.max_lr, args.lr)
logger.debug('init training number of iters '+str(n_iters)+' and batches '+str(n_batches))
m.inference.initialize(n_samples=1, n_iter=n_iters, logdir=m.logdir,
                       scale={m.y_pos: n_batches, m.y_neg: n_batches / args.ns},
                       #kl_scaling={m.y_pos: kl_scaling_weights, m.y_neg: kl_scaling_weights},
                       optimizer=AdamOptimizer(learning_rate=m.learning_rate_placeholder)
                       )
early_stopping = EarlyStopping(patience=args.patience)
init = tf.global_variables_initializer()
sess.run(init)
logger.debug('....starting training')
iteration = 0
for epoch in range(args.n_epochs):
    d.batch = d.batch_generator(args.mb, args.noise)
    for batch in range(n_batches):
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
    m.saver.save(sess, os.path.join(m.logdir, "model.ckpt"), iteration)
    sigmas = m.sigU.eval()[:, 0]
    sigmas_list.append(sigmas)
    pickle.dump(sigmas_list, open(dir_name + "/sigmas.dat", "wb+"))
    if early_stopping.is_early_stopping(get_distance(sigmas)):
        break

logger.debug('training finished after '+str(epoch)+' epochs. Results are saved in ' + dir_name)
m.dump(dir_name + "/variational.dat", d)

logger.debug('Done')
