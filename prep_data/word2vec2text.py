from args import parse_args_word2vec2text
from utils import read_word2vec_embeddings


def main(word2vec_file, output):
    word2vec = read_word2vec_embeddings(word2vec_file)
    word2vec.save_word2vec_format(output, binary=False)


if __name__ == '__main__':
    args = parse_args_word2vec2text()
    main(word2vec_file=args.word2vec_file, output=args.out_file)