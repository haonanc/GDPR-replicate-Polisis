# Neal Haonan Chen (hc4pa)
# University of Virginia
# Tensorflow Implementation of the Multilabel CNN used in Polisis text classfication.

import util
import train

if __name__ == "__main__":
    (train_texts, train_labels), (val_texts, val_labels) = (util.load_gdpr_dataset('samples.txt','labels.txt'))
    util.get_num_words_per_sample(train_texts)
    train_texts,val_texts,word_index = util.sequence_vectorize(train_texts,val_texts,top_k=20000,max_seq_len=500)
    embedding_matrix = util.load_embedding_matrix(word_index,100)
    model = train.train_sequence_model(train_texts=train_texts,
                               test_texts=val_texts,
                               train_labels=train_labels,
                               test_labels=val_labels,
                               word_index=word_index,
                               dense_layer_size=100,
                               embedding_matrix=embedding_matrix,
                               num_classes=17,
                               learning_rate=0.01,
                               epochs=500,
                               batch_size=40,
                               filters = 200,
                               embedding_dim= 100,
                               kernel_size= 3,
                               pool_size= 1,
                               top_k= 20000,
                                       load_saved_weights="saved2.h5"
                                       )
    #load_saved_weights="saved.h5" Add this parameter if you want to load a saved h5 file, instead of training a new one
    preds = model.predict(val_texts)
    util.eval(preds,val_labels,threshold = 0.15)