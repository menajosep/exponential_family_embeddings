from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from args import *
from data import *
from bayesian_models import *


def get_empiric_probs(word_sampling_indexes, reverse_dictionary, counter):
    raw_empiric_probs = dict()
    empiric_probs = dict()
    for target, ctxt_array in word_sampling_indexes.items():
        for ctx in ctxt_array:
            ctx_count = counter[reverse_dictionary[ctx]]
            if (target, ctx) not in empiric_probs:
                raw_empiric_probs[(target, ctx)] = 1
            else:
                raw_empiric_probs[(target, ctx)] += 1
            empiric_probs[(target, ctx)] = 1. * raw_empiric_probs[(target, ctx)] / ctx_count
    return empiric_probs


if __name__ == "__main__":
    logger = get_logger()

    args, dir_name = parse_args_bayesian_test()
    os.makedirs(dir_name)
    logger.debug('Load data')
    d = pickle.load(open(args.in_file, "rb+"))
    logger.debug('get pos probs')
    pos_empiric_probs = get_empiric_probs(d.positive_word_sampling_indexes, d.reverse_dictionary, d.counter)
    logger.debug('get neg probs')
    neg_empiric_probs = get_empiric_probs(d.negative_word_sampling_indexes, d.reverse_dictionary, d.counter)
    logger.debug('dump probs')
    pickle.dump(pos_empiric_probs, open(dir_name + "/pos_empiric_probs.dat", "wb+"))
    pickle.dump(neg_empiric_probs, open(dir_name + "/neg_empiric_probs.dat", "wb+"))

    logger.debug('Done')
