import argparse
import os
import time
def parse_args():
        parser = argparse.ArgumentParser(description="run exponential family embeddings on text")

        parser.add_argument('--K', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--L', type=int, default=15000,
                            help='Vocabulary size. Default is 15000.')

        parser.add_argument('--n_iter', type=int, default = 100,
                            help='Number iterations. Default is 500.')

        parser.add_argument('--n_epochs', type=int, default=10,
                            help='Number of epochs. Default is 10.')

        parser.add_argument('--cs', type=int, default=4,
                            help='Context size. Default is 4.')

        parser.add_argument('--ns', type=int, default=20,
                            help='Number of negative samples. Default is 20.')

        parser.add_argument('--mb', type=int, default=5000,
                            help='Minibatch size. Default is 5000.')

        parser.add_argument('--sig', type=int, default=10.0,
                            help='Prior variance (regulariztion).')

        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate. Omitted if clr is used. Default is 0.01.')

        parser.add_argument('--in_file', type=str, default=None,
                            help='input file')

        parser.add_argument('--emb_type', type=str, default=None,
                            help='type of previously trained embeddings')

        parser.add_argument('--word2vec_file', type=str, default=None,
                            help='word2vec previously trained embeddings')

        parser.add_argument('--glove_file', type=str, default=None,
                            help='glove previously trained embeddings')

        parser.add_argument('--fasttext_file', type=str, default=None,
                            help='fasttext previously trained embeddings')

        parser.add_argument('--custom_file', type=str, default=None,
                            help='custom previously trained embeddings')

        parser.add_argument('--clr_type', type=str, default=None,
                            help='type of cyclic learning rate: triangular|triangular2|exp_range',
                            required=False,
                            choices=['triangular', 'triangular2', 'exp_range'])

        parser.add_argument('--base_lr', type=float, default=None,
                            help='low bound lr for cyclic learning rate',
                            required=False)

        parser.add_argument('--max_lr', type=float, default=None,
                            help='high bound lr for cyclic learning rate',
                            required=False)

        parser.add_argument('--clr_cycles', type=int, default=2,
                            help='Number of cycles for cyclic learning rate',
                            required=False)

        parser.add_argument('--patience', type=int, default=15,
                            help='Number of epochs to wait for improvements before stopping',
                            required=False)

        args =  parser.parse_args()
        dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

        return args, dir_name
