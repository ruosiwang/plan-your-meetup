from keras.layers import Dense, GlobalMaxPooling1D, Dropout, CuDNNGRU, GRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Sequential
from keras.initializers import Constant


def build_DNN(embedding_matrix, max_sequence_length,
              category_num, model_type=None):
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)

    if model_type is 'MLP':
        model = _MLP(model)
    if model_type is 'GRU':
        model = _GRU(model)
    if model_type is 'CNN':
        model = _CNN(model)
    if model_type is "CNN_GRU":
        model = _CNN_GRU(model)

    model.add(Dense(category_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def _MLP(model, layer_num=2):
    model.add(Flatten())

    for _ in range(layer_num):
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

    return model


def _CNN(model, layer_num=2):
    for i in range(layer_num):
        model.add(Conv1D(128, 5, activation='relu'))
        if i == layer_num - 1:
            model.add(GlobalMaxPooling1D())
        else:
            model.add(MaxPooling1D(5))
        model.add(Dropout(0.2))
    return model


def _GRU(model, layer_num=1):
    for _ in range(layer_num):
        model.add(GRU(100))
        model.add(Dropout(0.2))
    return model


def _CNN_GRU(model, layer_num=[1, 1]):
    cnn_lN, gru_lN = layer_num
    for _ in range(layer_num[0]):
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Dropout(0.2))

    for _ in range(layer_num[1]):
        model.add(GRU(100))
        model.add(Dropout(0.2))
    return model


def create_embedding_layer(embedding_matrix, max_sequence_length):
    return Embedding(embedding_matrix.shape[0],
                     embedding_matrix.shape[1],
                     embeddings_initializer=Constant(embedding_matrix),
                     input_length=max_sequence_length,
                     trainable=False)


def build_CNN(embedding_matrix, max_sequence_length, category_num):
    # create embedding layer
    embedding_layer = create_embedding_layer(embedding_matrix,
                                             max_sequence_length)

    model = Sequential()
    model.add(embedding_layer)

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(category_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def build_MLP(embedding_matrix, max_sequence_length, category_num):
    # create embedding layer
    embedding_layer = create_embedding_layer(embedding_matrix,
                                             max_sequence_length)

    model = Sequential()

    model.add(embedding_layer)
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(category_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def build_GRU(embedding_matrix, max_sequence_length, category_num):
    # create embedding layer
    embedding_layer = create_embedding_layer(embedding_matrix,
                                             max_sequence_length)

    model = Sequential()
    model.add(embedding_layer)

    model.add(CuDNNGRU(100))
    model.add(Dropout(0.2))

    model.add(Dense(category_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def build_CNN_GRU(embedding_matrix, max_sequence_length, category_num):
    embedding_layer = create_embedding_layer(embedding_matrix,
                                             max_sequence_length)

    model = Sequential()
    model.add(embedding_layer)

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))

    model.add(CuDNNGRU(100))
    model.add(Dropout(0.2))

    model.add(Dense(category_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model
