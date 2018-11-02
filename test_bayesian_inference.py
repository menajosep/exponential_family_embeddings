from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from args import *
from data import *
from bayesian_models import *

MAX_SAMPLES_PER_WORD = 10000


def batch_generator(word, d, pos_empiric_probs_dict, neg_empiric_probs_dict, pos_context_probs_dict, neg_context_probs_dict):
    word_samples = []
    pos_empiric_probs, neg_empiric_probs = [], []
    pos_ctxt_probs, neg_ctxt_probs = [], []
    word_index = d.dictionary[word]
    positive_word_sampling_indexes = d.positive_word_sampling_indexes[word_index]
    negative_word_sampling_indexes = d.negative_word_sampling_indexes[word_index]
    for positive_word_sampling_index in positive_word_sampling_indexes[:MAX_SAMPLES_PER_WORD]:
        word_samples.append((word_index, positive_word_sampling_index, 1))
        pos_empiric_probs.append(pos_empiric_probs_dict[(word_index, positive_word_sampling_index)])
        pos_ctxt_probs.append(pos_context_probs_dict[positive_word_sampling_index])
    for negative_word_sampling_index in negative_word_sampling_indexes[:MAX_SAMPLES_PER_WORD]:
        word_samples.append((word_index, negative_word_sampling_index, 0))
        neg_empiric_probs.append(neg_empiric_probs_dict[(word_index, negative_word_sampling_index)])
        neg_ctxt_probs.append(neg_context_probs_dict[negative_word_sampling_index])
    #shuffle(word_samples)
    target_words, context_words, labels = zip(*word_samples)
    labels = np.array(labels)
    word_target = np.array(target_words, dtype="int32")
    word_context = np.array(context_words, dtype="int32")
    return word_target, word_context, labels, pos_empiric_probs, neg_empiric_probs, pos_ctxt_probs, neg_ctxt_probs


def run_tensorflow(word, d, pos_empiric_probs, neg_empiric_probs):
    if len(d.positive_word_sampling_indexes[d.dictionary[word]]) > 0 and len(
            d.negative_word_sampling_indexes[d.dictionary[word]]) > 0:
        logger.debug('....starting predicting')
        target_words, context_words, labels, pos_empiric_probs, neg_empiric_probs, pos_ctxt_probs, neg_ctxt_probs = \
            batch_generator(word, d, pos_empiric_probs, neg_empiric_probs, pos_context_probs, neg_context_probs)

        perplexity_pos, perplexity_neg = sess.run([m.perplexity_pos, m.perplexity_neg], {
            m.target_placeholder: target_words,
            m.context_placeholder: context_words,
            m.labels_placeholder: labels,
            m.batch_size: len(labels),
            m.pos_empiric_probs: pos_empiric_probs,
            m.neg_empiric_probs: neg_empiric_probs,
            m.pos_ctxt_probs: pos_ctxt_probs,
            m.neg_ctxt_probs: neg_ctxt_probs
        })
        return perplexity_pos, perplexity_neg
    else:
        return -1, -1


def get_empiric_probs(word_sampling_indexes):
    raw_empiric_probs = dict()
    empiric_probs = dict()
    ctxt_counter = dict()
    for target, ctxt_array in word_sampling_indexes.items():
        for ctx in ctxt_array:
            if ctx not in ctxt_counter:
                ctxt_counter[ctx] = 1
            else:
                ctxt_counter[ctx] += 1
            if (target, ctx) not in empiric_probs:
                raw_empiric_probs[(target, ctx)] = 1
            else:
                raw_empiric_probs[(target, ctx)] += 1
    for pair, count in raw_empiric_probs.items():
        empiric_probs[pair] = count / ctxt_counter[pair[1]]

    context_probs = {k: v / sum(ctxt_counter.values()) for k, v in ctxt_counter.items()}

    return empiric_probs, context_probs


if __name__ == "__main__":
    logger = get_logger()

    args, dir_name = parse_args_bayesian_test()
    os.makedirs(dir_name)
    logger.debug('Load data')
    d = pickle.load(open(args.in_file, "rb+"))
    logger.debug('get pos probs')
    pos_empiric_probs, pos_context_probs = get_empiric_probs(d.positive_word_sampling_indexes)
    logger.debug('get neg probs')
    neg_empiric_probs, neg_context_probs = get_empiric_probs(d.negative_word_sampling_indexes)
    logger.debug('Load embeddings')
    d.load_embeddings(args.emb_type, args.word2vec_file, args.glove_file,
                      args.fasttext_file, None, logger)
    if args.shuffle:
        aux_embs = d.embedding_matrix.copy()
        np.random.shuffle(aux_embs)
        d.embedding_matrix = aux_embs

    sigmas = None
    if args.sigmas:
        sigmas_array = pickle.load(open(args.sigmas, "rb+"))
        sigmas = sigmas_array[-1]

    pos_perplexities = dict()
    neg_perplexities = dict()
    sess = ed.get_session()

    # MODEL
    logger.debug('....build model')
    m = bayesian_emb_inference_model(d, sess, dir_name, sigmas)

    # INFERENCE
    init = tf.global_variables_initializer()
    sess.run(init)
    for word in sorted(d.dictionary):
        logger.debug('predicting ' + word)
        local_pos_perplexity = []
        local_neg_perplexity = []
        for i in range(args.n_samples):
            pos, neg = run_tensorflow(word, d, pos_empiric_probs, neg_empiric_probs)
            local_pos_perplexity.append(pos)
            local_neg_perplexity.append(neg)
        local_pos_perplexity = np.array(local_pos_perplexity)
        local_neg_perplexity = np.array(local_neg_perplexity)
        pos_perplexities[word] = {
            "max" : local_pos_perplexity.max(),
            "min": local_pos_perplexity.min(),
            "loc": local_pos_perplexity.mean(),
            "std": local_pos_perplexity.std(),
        }
        neg_perplexities[word] = {
            "max": local_neg_perplexity.max(),
            "min": local_neg_perplexity.min(),
            "loc": local_neg_perplexity.mean(),
            "std": local_neg_perplexity.std(),
        }
    logger.debug('Store data')
    pickle.dump(pos_perplexities, open(dir_name + "/pos_perplexities.dat", "wb+"))
    pickle.dump(neg_perplexities, open(dir_name + "/neg_perplexities.dat", "wb+"))

    logger.debug('Done')
