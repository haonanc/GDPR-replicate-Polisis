# Neal Haonan Chen (hc4pa)
# University of Virginia
# Tensorflow Implementation of a Multilabel CNN used in Polisis text classfication.


import os
import copy
import random
import numpy as np
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

import matplotlib.pyplot as plt

def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

def load_gdpr_dataset(data_path,label_path,train_percent = 0.8,seed = 0):

    label_list = []
    with open(label_path) as label_file:
        for line in label_file:
            label_list.append(line.strip("\n"))
    print(label_list)
    data_path = os.path.join(data_path)
    text = []
    labels = []
    odd = True
    with open(data_path) as infile:
        for line in infile:
            if odd:
                labels.append(labelBinarize(label_list,line.split()))
                odd = False
            else:
                odd = True
                text.append(line.strip("\n"))

    random.seed(seed)
    random.shuffle(text)
    random.seed(seed)
    random.shuffle(labels)
    divide = int(len(text)*train_percent)
    train_text = text[:divide]
    train_labels = labels[:divide]
    test_text = text[divide:]
    test_labels = labels[divide:]
    return (train_text,np.array(train_labels)),(test_text,np.array(test_labels))

def labelBinarize(label_list,labels):
    ret = [0]*len(label_list)
    for label in labels:
        ret[label_list.index(label)] = 1
    return ret

def sequence_vectorize(train_text,val_text, top_k = 20000, max_seq_len = 500):
    tokenizer = text.Tokenizer(num_words=top_k)
    tokenizer.fit_on_texts(train_text)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_text)
    x_val = tokenizer.texts_to_sequences(val_text)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > max_seq_len:
        max_length = max_seq_len

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

def load_embedding_matrix(word_index,embedding_dim):
    embeddings_index = {}
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Found %s word vectors.' % len(embeddings_index))
    return embedding_matrix

def eval(preds_,labels_,threshold = 0.25):
    threshold = 0.2
    for i in range(0,25):
        preds = copy.deepcopy(preds_)
        labels = copy.deepcopy(labels_)
        threshold += 0.025
        miss = 0
        perfect = 0
        preds[preds >= threshold] = int(1)
        preds[preds < threshold] = int(0)
        for i in range(len(preds)):
            matrix = preds[i]-labels[i]
            matrix = np.absolute(matrix)
            if matrix.sum() == 0:
                perfect +=1
            miss += matrix.sum()**2
            # positive 1 means a missed, negative 1 means a false positive
            if i == 500:
                break
        print("threshold:",threshold,",Perfect Count:",perfect,",Missed^2:",miss/500)
    return