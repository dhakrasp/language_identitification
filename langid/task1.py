import os
from collections import defaultdict
from pathlib import Path
from string import punctuation

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Bidirectional, LSTM, Dense, Dropout, Embedding,
                          Input)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.set_random_seed(2)

data_dir = Path('./langid-data-small/task1')


def get_raw_data(data_dir):
    data_dict = {}
    for fname in os.listdir(data_dir):
        fpath = data_dir / fname
        with open(fpath, 'r', encoding='utf-8') as file:
            temp = list(map(lambda x: x.strip(), file.readlines()))
        data_dict[fpath.suffix.lstrip('.')] = temp
    return data_dict


def text_to_ids(text, vocab):
    return [vocab[c] for c in text]


def get_vectorized_data(data_dict, vocab, label2id):
    Y = []
    sequences = []
    for lang, data in data_dict.items():
        for text in data:
            sequences.append(text_to_ids(text, vocab))
            Y.append(label2id[lang])
    X = pad_sequences(sequences, maxlen=200, dtype='int32', padding='post', truncating='post', value=0)
    Y = np.array(Y)
    return X, Y


def make_model(input_dim, embed_dim, hidden_dim, output_dim):
    dropout_prob = 0.
    model = Sequential()
    model.add(Embedding(input_dim, embed_dim, embeddings_initializer='uniform'))

    r1, r2 = None, None
    # r1=regularizers.l2(0.001)
    # r2=regularizers.l2(0.001)

    model.add(Bidirectional(LSTM(units=hidden_dim, kernel_regularizer=r1), merge_mode='concat'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(output_dim, activation='softmax', kernel_regularizer=r2))
    print(model.summary())
    optim = Adam(lr=1e-3, clipnorm=5., clipvalue=1.)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], optimizer=optim)
    return model


if __name__ == "__main__":
    # get the raw texts for each language
    data_dict = get_raw_data(data_dir)

    # prepare the vocabulary of characters
    vocab = defaultdict(int)
    for lang, data in data_dict.items():
        for line in data:
            for char in line:
                if char not in vocab:
                    vocab[char] = len(vocab) + 1

    # create a language to idx map
    label2id = {k: idx for idx, k in enumerate(data_dict.keys())}
    print(len(vocab))
    print(label2id)

    '''
        vectorize the data into numpy tensors
        X: 2D tensor
        Y: 1D tensor
    '''
    X, Y = get_vectorized_data(data_dict, vocab, label2id)
    print(X.shape, Y.shape)
    input_dim, output_dim = len(vocab) + 1, len(label2id)
    embed_dim, hidden_dim = 100, 100
    model = make_model(input_dim, embed_dim, hidden_dim, output_dim)
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = str(model_dir/'first')

    ckpt_clbk = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_clbk = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
    callbacks = [ckpt_clbk, early_clbk]
    model.fit(x=X, y=Y, batch_size=32, epochs=50, validation_split=0.2, callbacks=callbacks)
