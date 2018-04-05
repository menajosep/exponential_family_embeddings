import pickle
import numpy as np
variational_datafile = '/Users/jose.mena/dev/personal/exponential_family_embeddings/fits/variational.dat'
variational_data = pickle.load(open(variational_datafile, 'rb'))
rhos = variational_data['rhos']
vocabulary_file = '/Users/jose.mena/dev/personal/exponential_family_embeddings/fits/vocab.tsv'

embeddings_file = '/Users/jose.mena/dev/personal/exponential_family_embeddings/embeddings.txt'
with open(embeddings_file, 'wv') as emb_file:
    with open(vocabulary_file, 'r') as vocab_file:
        i = 0
        for word_line in vocab_file.readlines():
            word = word_line.replace('\n','')
            rho = rhos[i]
            nomr_rho = rho / np.linalg.norm(rho)
            rho_string = np.array2string(nomr_rho, separator=' ', max_line_width=100)[1:]
            rho_string = rho_string[:len(rho_string)-1].replace('\n','')
            emb_file.write(word+ ' ' +rho_string + '\n')
            i += 1
