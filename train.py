# Neal Haonan Chen (hc4pa)
# University of Virginia
# Tensorflow Implementation of a Multilabel CNN used in text classfication.


import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.optimizers import SGD



def get_model(dense_layer_size,
                 filters,
                 kernel_size,
                 embedding_dim,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 embedding_matrix=None):
    """
    Create the CNN architecture used by Polisis
    :param dense_layer_size: dense layer size
    :param filters: size of the filters
    :param kernel_size: size of the kernel
    :param embedding_dim: number of dimensions for the embedding layers
    :param pool_size: pooling size
    :param input_shape: input shape
    :param num_classes: number of labels for output
    :param num_features: number of features for input of embedding layers, default to 500
    :param embedding_matrix: load the embedding
    :return: CNN model
    """
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=False))
    # model.add(Dropout(rate=0.1))
    model.add(SeparableConv1D(filters=filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='valid'))
    # model.add(SeparableConv1D(filters=filters,
    #                           kernel_size=kernel_size,
    #                           activation='relu',
    #                           bias_initializer='random_uniform',
    #                           depthwise_initializer='random_uniform',
    #                           padding='valid'))
    # model.add(GlobalAveragePooling1D())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()
    return model


def train_sequence_model(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         word_index,
                         dense_layer_size,
                         embedding_matrix,
                         num_classes,
                         learning_rate,
                         epochs,
                         batch_size,
                         filters,
                         embedding_dim,
                         kernel_size,
                         pool_size,
                         top_k,
                         load_saved_weights = None):
    """
    #TODO implement the history module
    :param train_texts: train_texts
    :param test_texts: test_texts
    :param train_labels: train_labels
    :param test_labels: test_labels
    :param word_index: word_index
    :param dense_layer_size: size of the output layer
    :param embedding_matrix: load a word embedding or not
    :param num_classes: number of classes
    :param learning_rate: learning rate
    :param epochs: max number of epochs
    :param batch_size: batch size
    :param filters: number of filters
    :param embedding_dim: embedding dimensions
    :param kernel_size: size of kernel
    :param pool_size: size of pooling layer
    :param top_k: top_k words
    :param load_saved_weights: training a new model or load a trained model
    :return: trained model
    """

    num_features = min(len(word_index) + 1, top_k)
    # Create model instance.
    model = get_model(dense_layer_size=dense_layer_size,
                      filters=filters,
                      kernel_size=kernel_size,
                      embedding_dim=embedding_dim,
                      pool_size=pool_size,
                      input_shape=train_texts.shape[1:],
                      num_classes=num_classes,
                      num_features=num_features,
                      embedding_matrix = embedding_matrix)

    if load_saved_weights != None:
        model.load_weights(load_saved_weights)
    else:
        optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss="binary_crossentropy")
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]
        history = model.fit(
                train_texts,
                train_labels,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=(test_texts, test_labels),
                verbose=2,  # Logs once per epoch.
        batch_size = batch_size)
        model.save('saved2.h5')

    return model
    # return history['val_acc'][-1], history['val_loss'][-1]