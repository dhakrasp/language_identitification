import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from string import punctuation

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.set_random_seed(2)

data_dir = Path('./langid-data-small/task1')


def get_raw_data(data_dir):
    data_dict = {}
    for fname in os.listdir(data_dir):
        fpath = data_dir / fname
        with open(fpath, 'r', encoding='utf-8') as file:
            temp = list(map(lambda x: x.lower().strip(), file.readlines()))
        data_dict[fpath.suffix.lstrip('.')] = temp
    return data_dict


def get_raw_texts(fpath):
    with open(fpath, 'r', encoding='utf-8') as file:
        return list(map(lambda x: x.lower().strip(), file.readlines()))


def text_to_ids(text, vocab):
    return [vocab[c] if c in vocab else 0 for c in text]


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


def get_vectorized_texts(texts, vocab):
    sequences = []
    for text in data:
        sequences.append(text_to_ids(text, vocab))
    X = pad_sequences(sequences, maxlen=200, dtype='int32', padding='post', truncating='post', value=0)
    return X


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
    model.add(Dropout(dropout_prob))
    print(model.summary())
    return model


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_dir', action='store', dest='model_dir', default='models',
                        help='Location to store the models')
    parser.add_argument('--model_file', action='store', dest='model_file', default='task_1.model',
                        help='Filename for the model')
    parser.add_argument('--vocab_file', action='store', dest='vocab_file', default='vocab.pkl',
                        help='Filename for the vocab')
    parser.add_argument('--labels_file', action='store', dest='labels_file', default='labels.pkl',
                        help='Filename for the labels')
    parser.add_argument('--train_path', action='store', dest='train_path', default='langid-data-small/task1/train',
                        help='Path to training data')
    parser.add_argument('--dev_path', action='store', dest='dev_path', default='langid-data-small/task1/train',
                        help='Path to test data')
    parser.add_argument('--test_path', action='store', dest='test_path', default='langid/langid.test',
                        help='Path to test data')
    parser.add_argument('--test_output_path', action='store', dest='test_output_path', default='langid/langid.test_labels',
                        help='Path to test outputs')
    parser.add_argument('--mode', action='store', dest='mode',
                        default='test',
                        help='Indicates the mode (train/test/predict)')
    opt = parser.parse_args()

    # model storage location
    model_dir = Path(opt.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = str(model_dir / opt.model_file)
    vocab_file = str(model_dir / opt.vocab_file)
    labels_file = str(model_dir / opt.labels_file)

    if opt.mode == 'train':
        # set the training data path
        data_dir = Path(opt.train_path)

        # get the raw texts for each language
        data_dict = get_raw_data(data_dir)

        # prepare the vocabulary of characters
        vocab = defaultdict(int)
        for lang, data in data_dict.items():
            for line in data:
                for char in line:
                    if char not in vocab:
                        vocab[char] = len(vocab) + 1

        save_pickle(vocab, vocab_file)
        print(f'Vocab size: {len(vocab)}')

        # create a language to idx map
        label2id = {k: idx for idx, k in enumerate(data_dict.keys())}
        save_pickle(label2id, labels_file)
        print(f'Labels:\n{label2id}')

        '''
            vectorize the data into numpy tensors
            X: 2D tensor
            Y: 1D tensor
        '''
        X, Y = get_vectorized_data(data_dict, vocab, label2id)

        # split into training and dev sets
        # X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2)
        print(f'Training data size: {X.shape[0]}')
        # print(f'Dev data size: {X_dev.shape[0]}')

        # set hyperparams
        input_dim, output_dim = len(vocab) + 1, len(label2id)
        embed_dim, hidden_dim = 100, 100

        # create model
        model = make_model(input_dim, embed_dim, hidden_dim, output_dim)

        # compile model
        optim = Adam(lr=1e-3, clipnorm=5., clipvalue=1.)
        model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], optimizer=optim)

        # callbacks for saving and early stopping
        ckpt_clbk = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_clbk = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
        callbacks = [ckpt_clbk, early_clbk]

        epochs = 50
        batch_size = 32
        model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks)

    elif opt.mode == 'test':
        # set the dev data path
        data_dir = Path(opt.dev_path)

        # get the raw texts for each language
        data_dict = get_raw_data(data_dir)
        print(data_dict['en'][:5])

        # load vocab, labels and model
        vocab = load_pickle(vocab_file)
        print(len(vocab))
        label2id = load_pickle(labels_file)
        model = load_model(model_file)

        # vectorize the dev data
        X, Y = get_vectorized_data(data_dict, vocab, label2id)

        # predict the languages
        y_pred = np.argmax(model.predict(X), axis=-1)

        print(label2id)
        # print metrics
        print(classification_report(Y, y_pred))
        print(confusion_matrix(Y, y_pred))

    elif opt.mode == 'predict':
        # get test data
        test_data = get_raw_texts(opt.test_path)

        vocab = load_json(vocab_file)
        label2id = load_json(labels_file)
        model = load_model(model_file)

        # vectorize test data
        X = get_vectorized_texts(test_data, vocab)

        # predict
        y_pred = np.argmax(model.predict(X), axis=-1)
        with open(opt.test_output_path, 'w') as file:
            for label in y_pred:
                file.write(f'{y_pred}\n')

    else:
        print('Incorrect operation mode. Valid values for mode are (train/test/predict)')
