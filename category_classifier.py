import dataset as d
import helper as h
import keras_models as k
import sklearn_models as s

from collections import Counter

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

DATA_TYPE = 'group'  # or Event
MODEL_GROUP = 'sklearn'  # or sklearn
CATEGORY_NUM = 15
EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
SEQ_MAXLEN = 1000

Keras_models = [('MLP', 2), ('GRU', 1), ('CNN', 3), ('CNN_GRU', [2, 1])]
sklearn_models = ['MNB', 'SVC', 'ETC', 'RFC']

# load data and prepare


def make_dataset(data_type='group', category_num=CATEGORY_NUM, sample_balance=False):
    print(f"prepare {data_type} dataset...")
    if data_type is 'group':
        descriptions, categories = d.load_group_descriptions()
    elif data_type is 'event':
        descriptions, categories = d.load_event_descriptions()

    # tranform categorical labels(str) to Ids(int) and sorted by popularity
    labels, mapping = d.label_transform(categories)

    # take the top N popoplar categories
    # balance the number of samples in each category
    descriptions, labels = d.prepare_dateset(descriptions, labels,
                                             category_num=category_num,
                                             sample_balance=False)
    # text cleaning
    descriptions = h.clean_texts(descriptions)

    # dataset descriptions:
    print(f'This dataset includes {len(descriptions)} group \
            descriptions on the following categories:')

    return descriptions, labels


# DATA PREPROCESSING FOR DEEP LEARNING
def data_transform(descriptions, labels, max_num_words, seq_maxlen):
    print("data preprocessing...")
    # tokenize group descriptions
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(descriptions)
    sequences = tokenizer.texts_to_sequences(descriptions)
    # padd sequences
    sequences = pad_sequences(sequences, maxlen=seq_maxlen)
    # one hot coding
    onehot_labels = to_categorical(labels)
    print("preprocessing done...")
    return sequences, onehot_labels, tokenizer.word_index


if __name__ == '__main__':
    descriptions, labels = make_dataset(data_type=DATA_TYPE,
                                        category_num=CATEGORY_NUM,
                                        sample_balance=False)

    if MODEL_GROUP is 'keras':
        sequences, onehot_labels, word_index = data_transform(descriptions, labels,
                                                              MAX_NUM_WORDS, SEQ_MAXLEN)
        # load glove embedding
        glove = h.load_glove(EMBEDDING_DIM)
        embedding_matrix = h.get_embedding_matrix(word_index, glove,
                                                  MAX_NUM_WORDS, EMBEDDING_DIM)

    # # train test split
    X_train, X_val, y_train, y_val = \
        train_test_split(descriptions, labels, test_size=0.2, random_state=42)

    if MODEL_GROUP is 'keras':
        for (model_type, layer_num) in Keras_models:
            print(f'{model_type} training')
            model_name = f'{model_type}_{CATEGORY_NUM:02d}_{DATA_TYPE}'
            model = k.build_DNN(embedding_matrix, SEQ_MAXLEN,
                                CATEGORY_NUM, model_type=model_type)
            callbacks = k.set_callback(model_name)
            model.fit(X_train, y_train, epochs=10,
                      batch_size=128, callbacks=callbacks,
                      validation_split=0.2)
            result = h.get_evaluations(model, X_val, y_val)

    if MODEL_GROUP is 'sklearn':
        for model_type in sklearn_models:
            print(f'{model_type} training')
            model_name = f'{model_type}_{CATEGORY_NUM:02d}_{DATA_TYPE}'
            model = s.build_SKM(model_type=model_type,
                                max_features=MAX_NUM_WORDS,
                                selectK=10000)
            model.fit(X_train, y_train)
            s.save_model(model, model_name)
            results = h.get_evaluations(model, X_val, y_val)
