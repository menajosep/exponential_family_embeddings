# Uncertainty-based Intrinsic Evaluation of Word Embeddings

## References

This code is based on the Bernoulli Embeddings as described in this paper:


[M. Rudolph, F. Ruiz, S. Mandt, D. Blei, **Exponential Family Embeddings**,
*Neural Information Processing Systems*, 2016](http://www.cs.columbia.edu/~blei/papers/RudolphRuizMandtBlei2016.pdf)

### Data sources
In this work we use 4 data sources:
* Recipes: The recipes box data set from https://eightportions.com/datasets/Recipes/ that compiles recipes from various recipes web sites.
* Wikipedia: data obtained from the First billion characters from wikipedia, http://mattmahoney.net/dc/enwik9.zip
* Economic news articles: from the Economic News Article Tone and Relevance, https://www.figure-eight.com/data-for-everyone/
* Computer science publications: obtained from the dataset described in the article "Deep Keyphrase Generation" included in the Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

## Purpose
The increasing number of methods to learn word embeddings makes necessary to find procedures for evaluating their quality, especially when applied to very specific domains. Validation mechanisms exist that evaluate the quality of those embeddings by applying them to auxiliary tasks. In the present work, we describe a validation method that is derived directly from the training of these embeddings and allows practitioners to evaluate their convenience to particular use cases based on a quantitative metric. By applying a Bayesian inference approach we obtain, not only an estimate for the embedding vectors, but also a probability distribution that provides a measure of the uncertainty of each embedding with no additional cost. By analyzing the distribution of these uncertainties we can deliver a measure of the quality of the embeddings. Moreover, we show that this method can be applied to pre-trained embeddings to measure their suitability for a concrete domain or corpus. The method is validated in different auxiliary tasks showing coherent results with the performance in those tasks.

## Method
### Preprocessing

In the folder *prep_data* there is a script for each data source. What these scripts do is read the raw data files and preprocess it by applying preprocess_text method of the **textacy** library. Basically, what it does is to remove emails, urls, phone numbers, numbers, contractions, punctuation symbols, currency symbols, accents, it fixes unicode issues, convert the text into lower case and transliterate non ascii characters into their closest ascii equivalents.

In this version of the scripts, the output is written with each sentence in a separate line, to ease the separation of contexts for each piece of texts. That's why the output file names are all suffixed with multiline.

Each data source file has different formats: csv files, json, dictionaries, so the treatment is different for all of them, but the output remains the same: a line per piece of text with the preprocessing method applied as described.

#### Decisions:
* The wikipedia dataset needs a previous preprocessing using the script https://github.com/facebookresearch/fastText/blob/master/wikifil.pl. In our case, we slightly modify the script with the `#!/usr/bin/perl -l` directive at the beginning of the script to have a separated line per article.
* The recipes, economics and CS papers datasets hold a separate line for the title and the body of the article. As there are some very short titles, it could be a good idea to join both fields into the same line as, despite belonging to different pieces of texts, they may share meanings, or maybe discard short texts with less words than the window size, for example.
* In some cases, especially for compound words, **textacy** breaks the word in two pieces, removing the hyphen. This may lead to error as some of these compound words exist in the dictionaries that we use later on.
* Another good idea could be to use a language detector, like **langdetect** to get only English texts, avoiding to handle non-English words and texts.

### Building the dictionary
In the data.py script, the class *bayessian_bern_emb_data* holds the logic that builds the dictionary and sample data. These are the parameters:
* input_file: input file with the raw data.
* cs: context size, or windows size. Size of the full window, so it must be multiple of 2.
* ns: number of negative samples.
* n_minibatch: size of the mini batch.
* L: Size of the dictionary.
* K: Size of the embedding vectors.
* emb_type: type of embedding to use for the vectors (glove|word2vec|fasttext|custom)
* word2vec_file: file that holds the embedding model for word2vec.
* glove_file: file that holds the embedding model for glove.
* fasttext_file: file that holds the embedding model for fasttext.
* custom_file: file that holds the embedding model for adhoc trained vectors.
* fake_sentences_number: number of fake sentences to add to the data set.
* dir_name: working directory.
* logger

After reading the raw data from the file, all sentences are stored in the sentences object and the different embedding models are stored as variables also. After it, the first thing is to build the dictionary of words:
1. Init a vector for a counter, and fill the first position with the symbol 'UNK' for the unknowns and the corresponding counter to -1.
1. Count all words in sentences and sort them (ascending) by the number of occurrences, taking only the first L-1 elements(remember that we already have a first element for the unknowns so they add up the L elements). If L >> ord(words), then L will be number of words + 1.
1. Build the dictionary by assigning the word as the key and the position in the ascending sorted counter as the value. This is done only for words that are also preent in all the vocabularies of the embedding models. Also build a parallel structure for the counting of words.
1. Fix L to the size of the built dictionary.
1. Build a reverse dictionary with the index as key and the word as values.
1. Finally, build another attribute in the class that holds a list of the words of the dictionary.

#### Decisions:
* Reviewer: *"To create a vocabulary of words that are present in all three pretrained embedding for fairness is ... not really fair. Models that tried to optimize the now out-of-vocabulary words are penalized. Why not pick the vocabulary of common words (as for now) but retrain all three models using only that vocabulary ? With the availability of Glove, Word2Vec and fastText implementations, this would be reasonably achievable, and really fair."*

### Build the samples
Once we built the dictionary, it's time to build the samples. This task is done in parallel, processing a sentence each time. For each sentence this is the process:
1. First the sentence is split into an array of words
1. We add a padding for half the size of the window at the beginning and at the end of the words array. We do this to be able to include the first and last words as target words in the sampling.
1. Then we filter and process only those sentence with more words than the size of the windows, to have enough words for the sampling.
1. Then we start processing each word in the array
  1. check that the word is in the dictionary and is not unknown.
  1. Get a random window size sampling from a uniform.
  1. Fix the target word as the index of the current word and context words for the indexes in -cs/2 to cs/2.
  1. If context words is in the dictionary and is not the padding symbol, add the pair target and context words to the positive samples.
  1. Finally, add NS negative samples by picking a random word from the dictionary each time.
Once we have all the positive and negative samples for all the sentences, we build two dictionaries, one for the positive and another for the negative samples. Each dictionary has an entry for every word in the dictionary, where the key is the word and the values correspond to the array of samples.

#### Decisions:
* The random size of the window is to mimic what is done in FastText.
* There might be words that won't have samples associated if they appear only in short sentences.

### Load the Embeddings
After building the dictionary and the samples, it's time to load the embedding matrix. First of all, according to the type of embedding in emb_type, we load the embedding model. Next, we initialize a matrix of size equal to the size of the dictionary to zeros. Next, we go through the elements of the dictionary and retrieve the vector corresponding to every word in the dictionary, and including it in the matrix of embeddings at the same index it has in the original dictionary. All words, except unknown will have a corresponding entry, as we already checked it when building the dictionary.

### Build the models

The bayesian_models.py file holds the bayesian_emb_model class that defines the model used to train the epistemic uncertainties. We call them epistemic as, at this point, what we want is to capture the uncertainty introduced by the embeddings that we are using, as part of the model in this case.

The model receives the data object built before and only an extra parameter that is the initial value for the sigma of the Normal distribution we use as priors for our inference method. From the data object we will use the variables:
* minibatch: size of the mini batch.
* L: size of the dictionary.
* K: size of the embedding vectors.
* embedding_matrix: matrix with the embedding vectors for all words in the dictionary.

The process of the definition of the model is as follows:
1. Definition of the placeholders. Data structures that will receive the input data. Theser are the inputs:
  1. target_placeholder: the target word indexes for each sample pair of the batch.
  1. context_placeholder: the context word indexes for each sample pair of the batch.
  1. labels_placeholder: label that tells whether it is a positive or a negative sample.
  1. ones_placeholder and zeros_placeholder: auxiliary vectors full of ones for positive and zeros for negative examples.
  1. learning_rate_placeholder: the learning rate to apply on each batch(related to cyclic learning rate implementation)
1. Define the priors for the probability distributions of the embedding matrix. These priors are based on a Normal ~ N(0,1).
1. Using the label, filter the positive examples and gather the corresponding embedding vectors for each word index, both for the target and context words.
1. Repeat the same process for the negative examples.
1. For positive examples, compute the inner product of the target and context embedding vectors. Repeat it for the negative samples.
1. Use the resulting coeficients as the logits to define a Bernoulli RV for positives and another one for the negatives.
1. Initialize the posterior conjugate of the embeddings with Normal variables where the location are the embedding vectors from the embedding matrix, and the scale is a variable we will train initialized with the sigma value and filtered with a softplus function to prevent negative values.
1. Use Variational inference, KLqp, to approximate the true posterior.

#### Decisions:
* The variable that we are learning in this case, the sigma of the distribution,is initialized using the same variable for all the positions of the vector. This is what is called a spherical, or isotropic, variance matrix. Thus the resulting model is simpler than if wanted to capture the variance for the components of the vectors, but still complex enough to capture the variance or uncertainty of the vectorial representation of each word.
* Other distributions could have been used to define the priors or the models.

### Prepare the batches
Now it's turn to run the inference to approximate the posterior and, thus, obtain the distributions for the embeddings that will be located around the pre-trained embeddings but with a variance learned in the training process.
The training is organized on epochs. On each epoch, we go through the dictionary, and for each word we randomly pick CS positive and NS negative samples from the sampling dictionaries built in the data object. Thus, we get to have even number of training examples for all the words.
Once we build all the examples for the given epoch, we shuffle them and we split them on batches.

#### Decisions:
* In the original skip gram and in the Rudolph's implementation, they use a random uniform sampling method to pick the words to include on each batch. They use a modified distribution based on the unigrams to handle the underrepresentation of rare words in front of very frequent ones. We chose to sample each word once per epoch to assure that all sigmas will be trained with the same number of samples, though it is truth that the variability of the training samples will be very different between rare and frequent words.

### Inference
Once we have trained our model, we end up with a distribution for the embedding of any word of the dictionary. We have built an inference model to test the prediction capability of the resulting model.
The inference test has the following parameters:
* in_file: The file that holds the data prepared during the training.
* shuffle: whether to shuffle the embeddings or not. This was introduced to test that the embeddings where predicting well the labels.
* sigmas: the sigma or ucnertainty associated to each entry in the dictionary.
* emb_type, word2vec_file, glove_file, fasttext_file: Configuration of the embeddign models.
* n_samples: number of samples to throw from the embeddings distribution.

After loading the data stored in the training phase and the embedding matrix, we built a model specific for the inference task, which was very similar to the original one, but without the variational inference part.
This is the definition of the inference model:
1. Definition of the placeholders. Data structures that will receive the input data. Theser are the inputs:
  1. target_placeholder: the target word indexes for each sample pair of the batch.
  1. context_placeholder: the context word indexes for each sample pair of the batch.
  1. labels_placeholder: label that tells whether it is a positive or a negative sample.
  1. batch size: size of the current batch
1. Define the priors for the probability distributions of the embedding matrix. These priors are based on a Normal with location as the embedding vectors from the embedding matrix, and the scale as the learned sigmas. In case we just want to try the original empbeddings predictability, we can pass an array of zeros.
1. Using the label, filter the positive examples and gather the corresponding embedding vectors for each word index, both for the target and context words.
1. Repeat the same process for the negative examples.
1. For positive examples, compute the inner product of the target and context embedding vectors. Repeat it for the negative samples.
1. Use the resulting coeficients as the logits to define a Bernoulli RV for positives and another one for the negatives.
1. Sample a probability from the Bernoullis.

The prediction is done word by word, which means that each prediction batch corresponds to all the occurrences of a single word in the training sample.
This process is repeated N times, as indicated in the n_samples argument. Thus, we obtain the standard deviation of the predictions obtained each time we sample the embeddings distribution.

#### Decisions:
* As there are some very frequent words with thousands of samples, I limited, due to memory constraints, the number of predictions for each word to 10k at the most.

### Evaluation of the uncertainties
After all the previous process, we end up with an uncertainty value associated to each entry of the vocabulary. In theory, a good embedding should have most of these uncertainties close to zero, with higher values for rare words or words which representation is ambiguous. Empirically, though, we observe that the distribution of the values of these uncertainties is closer to a Beta distribution.
In our method, we take advantage of this expected behavior to formulate a metric for the quality of the embeddings, measuring how close is distribution of an actual embedding set to these optimal distributions.
The process of the definition of the metric is as follows:
1. Create a histogram with the values of the sigmas. The number and size of the bins is fixed to 100, and goes from 0 to 2, and it's shared across evaluations.
1. We define a family of Beta distributions, close to the ideal Dirac delta function.
1. We compute the KL divergence of the histogram of the sigmas with these Beta references and average the result.
This coefficient is then used to compare the quality if the embeddings, being the lower the better.
