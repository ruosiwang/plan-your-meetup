import dataset as d
import helper as h
import keras_models as k

import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

CATEGORY_NUM = 5
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 40000
SEQ_MAXLEN = 1000

print('HAHA')
# # ------------ DATASET PREPARTION --------------------
# # load data
# descriptions, categories = d.load_group_descriptions()
# # tranform categorical labels(str) to Ids(int) and sorted by popularity
# labels, mapping = d.label_transform(categories)
# # take the top N popoplar categories
# # balance the number of samples in each category
# descriptions, labels = d.prepare_dateset(descriptions, labels,
#                                          category_N=CATEGORY_NUM,
#                                          sample_balance=False)

# # ------------ DATA PREPROCESSING --------------------
# # text cleaning
# descriptions = h.clean_texts(descriptions)
# # tokenize group descriptions
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizer.fit_on_texts(descriptions)
# sequences = tokenizer.texts_to_sequences(descriptions)
# # padd sequences
# sequences = pad_sequences(sequences, maxlen=SEQ_MAXLEN)

# # one hot coding
# onehot_labels = to_categorical(labels)

# # train test split
# X_train, X_test, y_train, y_test = \
#     train_test_split(sequences, onehot_labels, test_size=0.2, random_state=42)

# # # ----------- WORD EMBEDDING --------------------------

# # load glove embedding
# glove = h.load_glove(EMBEDDING_DIM)
# embedding_matrix = h.get_embedding_matrix(tokenizer.word_index, glove,
#                                           MAX_NUM_WORDS, EMBEDDING_DIM)

# pickle.dump([X_train, X_test, y_train, y_test, embedding_matrix],
#             open("tmp.pkl", "wb"))

[X_train, X_test, y_train, y_test, embedding_matrix] = \
    pickle.load(open("tmp.pkl", "rb"))

# Conv NN
cnn = k.build_DNN(embedding_matrix, SEQ_MAXLEN, CATEGORY_NUM, model_type="CNN_GRU")
cnn.fit(X_train, y_train, batch_size=128, epochs=5,
        validation_split=0.2)
