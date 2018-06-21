import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import pickle
import numpy as np
import csv
from numpy import linalg as LA

variational_data = pickle.load(open('fits/remotefits/recipesall_fasttext/variational.dat', 'rb'))
embeddings = variational_data['rhos']

norm_embs = []
for emb in embeddings:
    norm_emb = emb/LA.norm(emb)
    norm_embs.append(norm_emb)
norm_embs = np.array(norm_embs)
playlist_sigmas = variational_data['sigma_rhos']
discard_noise_indexes = np.where(playlist_sigmas > 0.4)
good_embeddings = np.delete(norm_embs, discard_noise_indexes, axis=0)

# type = 'raw'|filtered
type = 'filtered'
if type == 'raw':
    embeddings = norm_embs
else:
    embeddings = good_embeddings


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embeddings.shape)
set_x = tf.assign(X, place, validate_shape=False)

sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embeddings})

out_dir = type
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
# write labels
with open(out_dir + '/'+type+'_metadata.tsv', 'w') as f:
    with open('fits/remotefits/recipesall_fasttext/vocab.tsv', 'r') as dict_file:
        dictionary_reader = csv.reader(dict_file, delimiter='\t')
        i = 0
        for row in dictionary_reader:
            if type == 'raw':
                line_out = "%s\n" % row[0]
                f.write(line_out)
            else:
                if not np.isin(i, discard_noise_indexes[0]):
                    line_out = "%s\n" % row[0]
                    f.write(line_out)
            i += 1

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(out_dir, '../'+type+'_metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
saver.save(sess, os.path.join(out_dir, type+"_model.ckpt"))