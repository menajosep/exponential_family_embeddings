import numpy as np
from utils import *
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams

class bern_emb_data():
    def __init__(self, cs, ns, n_minibatch, L):
        assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        #url = 'http://mattmahoney.net/dc/'
        #filename = maybe_download(url, 'text8.zip', 31344016)
        #filename = '/Users/jose.mena/Dropbox/Jose/PhD/data/recipes.txt.zip'
        filename = '../data/wiki/wiki.txt.zip'
        words = read_data(filename)
        self.build_dataset(words)
        self.batch = self.batch_generator()
        self.N = len(self.data)

    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.L - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
              index = dictionary[word]
            else:
              index = 0 
              unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data = np.array(data)
        self.count = count
        self.dictionary = dictionary
        self.words = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        sampling_table = sequence.make_sampling_table(len(dictionary))
        couples, labels = skipgrams(data,
                                    len(dictionary),
                                    window_size=self.cs,
                                    sampling_table=sampling_table,
                                    negative_samples=self.ns)
        del data
        self.labels = np.array(labels)
        # labels[labels == 0] = -1
        word_target, word_context = zip(*couples)
        del couples
        self.word_target = np.array(word_target, dtype="int32")
        self.word_context = np.array(word_context, dtype="int32")
        with open('fits/vocab.tsv', 'w') as txt:
            for word in self.words:
                txt.write(word+'\n')

    def batch_generator(self):
        batch_size = self.n_minibatch
        data_target = self.word_target
        data_context = self.word_context
        data_labels = self.labels
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, self.word_target])
                data_context = np.hstack([data_context, self.word_context])
                data_labels = np.hstack([data_labels, self.labels])
                if data_target.shape[0] < batch_size:
                    continue
            words_target = data_target[:batch_size]
            words_context = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield words_target, words_context, labels
    
    def feed(self, target_placeholder, context_placeholder, labels_placeholder, y_ph):
        words_target, words_context, labels = self.batch.next()
        return {target_placeholder: words_target,
                context_placeholder: words_context,
                labels_placeholder: labels,
                y_ph: np.ones((self.n_minibatch), dtype=np.int32)
                }

