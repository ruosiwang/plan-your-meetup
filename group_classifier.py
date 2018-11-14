import dataset as d
import helper as h
# import keras_models as km
import sklearn_models as sm


from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical

CATEGORY_NUM = 20
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 40000
SEQ_MAXLEN = 1000
Keras_models = [('MLP', 1)]
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
    return sequences, onehot_labels, tokenizer.word_index


if __name__ == '__main__':
    descriptions, labels = make_dataset(data_type='event',
                                        category_num=CATEGORY_NUM,
                                        sample_balance=False)

    # sequences, onehot_labels, word_index = data_transform(descriptions, labels,
    #                                                       MAX_NUM_WORDS, SEQ_MAXLEN)

    # # train test split
    X_train, X_val, y_train, y_val = \
        train_test_split(descriptions, labels, test_size=0.2, random_state=42)

    # # load glove embedding
    # glove = h.load_glove(EMBEDDING_DIM)
    # embedding_matrix = h.get_embedding_matrix(word_index, glove,
    #                                           MAX_NUM_WORDS, EMBEDDING_DIM)

    # for (model_type, layer_num) in Keras_models:
    #     model_name = f"{model_type}_{layer_num}"
    #     model = k.build_DNN(embedding_matrix, SEQ_MAXLEN, CATEGORY_NUM, model_type=model_type)
    #     model.fit(X_train, y_train, batch_size=128, epochs=5,
    #               validation_split=0.2)
    #     result = h.get_evaluations(model, X_test, y_test)
    bechmark = []
    for m in sklearn_models[:1]:
        model = sm.build_SKM(model_type=m,
                             max_features=MAX_NUM_WORDS,
                             selectK=10000)
        model.fit(X_train, y_train)
        results = h.get_evaluations(model, X_val, y_val)
        bechmark.append(results)
    print(results)
