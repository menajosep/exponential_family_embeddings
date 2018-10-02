from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from args import *
from data import *
from bayesian_models import *
import multiprocessing



def batch_generator(word, d):
    word_samples = []
    word_index = d.dictionary[word]
    positive_word_sampling_indexes = d.positive_word_sampling_indexes[word_index]
    negative_word_sampling_indexes = d.negative_word_sampling_indexes[word_index]
    for positive_word_sampling_index in positive_word_sampling_indexes:
        word_samples.append((word_index, positive_word_sampling_index, 1))
    for negative_word_sampling_index in negative_word_sampling_indexes:
        word_samples.append((word_index, negative_word_sampling_index, 0))
    shuffle(word_samples)
    target_words, context_words, labels = zip(*word_samples)
    labels = np.array(labels)
    word_target = np.array(target_words, dtype="int32")
    word_context = np.array(context_words, dtype="int32")
    return word_target, word_context, labels


def run_tensorflow(word):
    if len(d.positive_word_sampling_indexes[d.dictionary[word]]) > 0 and len(
            d.negative_word_sampling_indexes[d.dictionary[word]]) > 0:
        sess = ed.get_session()
        target_words, context_words, labels = batch_generator(word, d)
        # MODEL
        logger.debug('....build model')
        m = bayesian_emb_inference_model(d, sess, dir_name, len(labels), sigmas)

        # INFERENCE
        init = tf.global_variables_initializer()
        sess.run(init)
        logger.debug('....starting predicting')

        pos, neg = sess.run([m.prob_pos, m.prob_neg], {
                m.target_placeholder: target_words,
                m.context_placeholder: context_words,
                m.labels_placeholder: labels
            })
        pos_probs[word] = pos
        neg_probs[word] = neg


if __name__ == "__main__":
    logger = get_logger()
    manager = multiprocessing.Manager()

    args, dir_name = parse_args_bayesian_test()
    os.makedirs(dir_name)
    logger.debug('Load data')
    d = pickle.load(open(args.in_file, "rb+"))
    logger.debug('Load embeddings')
    d.load_embeddings(args.emb_type, args.word2vec_file, args.glove_file,
                      args.fasttext_file, None, logger)
    logger.debug('Shuffle:'+args.shuffle)
    if args.shuffle:
        aux_embs = d.embedding_matrix.copy()
        shuffle(aux_embs)
        d.embedding_matrix = aux_embs

    logger.debug('Sigmas:'+args.sigmas)
    sigmas = None
    if args.sigmas:
        sigmas_array = pickle.load(open(args.sigmas, "rb+"))
        sigmas = sigmas_array[-1]

    pos_probs = manager.dict()
    neg_probs = manager.dict()
    for word in d.dictionary:
        logger.debug('predicting '+word)
        # option 1: execute code with extra process
        p = multiprocessing.Process(target=run_tensorflow, args=[word])
        p.start()
        p.join()
    logger.debug('Store data')
    pickle.dump(pos_probs, open(dir_name + "/pos_probs.dat", "wb+"))
    pickle.dump(neg_probs, open(dir_name + "/neg_probs.dat", "wb+"))

    logger.debug('Done')
