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



# 2.Convolutional Layer, applies 200 3X300 filters(extracting 300-dimension vectors for 3 words) output 200 vectors of 1X(size of words - 2).
def get_model(dense_layer_size,
                 filters,
                 kernel_size,
                 embedding_dim,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 embedding_matrix=None):
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
    """Trains sequence model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
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
        model.save('saved.h5')




    return model
    # return history['val_acc'][-1], history['val_loss'][-1]