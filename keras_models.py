from keras.layers import Dense, GlobalMaxPooling1D, Dropout, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Sequential
from keras.initializers import Constant


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


def build_LSTM(embedding_matrix, max_sequence_length, category_num):
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


def build_CNN_LSTM(embedding_matrix, max_sequence_length, category_num):
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