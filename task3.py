import csv
import os
import pickle
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from string import punctuation

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          Input, Lambda, TimeDistributed)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# set seeds for consistent results
np.random.seed(1)
tf.set_random_seed(2)


def get_raw_data(fpath):
    sequences = []
    gold_labels = []
    with open(fpath, 'r', encoding='utf-8') as file:
        tweet_id = None
        seq = []
        labs = []
        for row in csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE):
            if int(row[0]) != tweet_id:
                tweet_id = int(row[0])
                if seq:
                    sequences.append(seq)
                    gold_labels.append(labs)
                    seq = []
                    labs = []
            seq.append(row[-2])
            labs.append(row[-1])
    return sequences, gold_labels


def get_raw_texts(fpath):
    """Fetches the raw texts for testing

    Arguments:
        fpath {str} -- Path to the test data file

    Returns:
        list -- List of strings. List of all the test samples
    """
    with open(fpath, 'r', encoding='utf-8') as file:
        return list(map(lambda x: x.lower().strip(), file.readlines()))


def text_to_ids(text, vocab):
    """Converts a text to a sequence of word ids

    Arguments:
        text {str} -- A data sample
        vocab {dict} -- A dict holding the word: idx mapping

    Returns:
        [list] -- List of integer ids
    """
    return [vocab[c] if c in vocab else 0 for c in text]


def get_vectorized_data(sequences, gold_labels, vocab, label2id, max_len, max_chars):
    """Converts text data and labels into tensors of integers

    Arguments:
        data_dict {dict} -- Dictionary with each language representing a key. The values are the corresponding (list of) training samples.
        vocab {dict} -- A dict holding the word: idx mapping
        label2id {dict} -- A dict holding the language: idx mapping

    Returns:
        np.array -- Rank 2 tensor holding all the vectorized training samples
        np.array -- Rank 1 tensor holding all the vectorized training labels
    """
    Y = []
    X = []
    for seq in sequences:
        for word in seq[:max_len]:
            char_seq = [vocab[c] if c in vocab else vocab['unk'] for c in word[:max_chars]]
            X.append(char_seq)
        for _ in range(max_len - len(seq)):
            X.append([0] * max_chars)
    X = pad_sequences(X, maxlen=max_chars, dtype='int32', padding='post', truncating='post', value=0)
    for lab_seq in gold_labels:
        vec = [label2id[lab] for lab in lab_seq[:max_len]] + [0] * max(0, max_len - len(lab_seq))
        one_hots = to_categorical(vec, num_classes=3)
        Y.append(one_hots)
    Y = np.array(Y)
    return np.reshape(X, newshape=(len(sequences), -1)), Y


def get_vectorized_texts(texts, vocab):
    """Converts text data into tensors of integers

    Arguments:
        texts {list} -- List of all the test samples. A list of strings
        vocab {dict} -- A dict holding the word: idx mapping

    Returns:
        np.array -- Rank 2 tensor holding all the vectorized samples
    """
    sequences = [text_to_ids(text, vocab) for text in texts]
    X = pad_sequences(sequences, maxlen=200, dtype='int32', padding='post', truncating='post', value=0)
    return X


def make_model(input_dim, embed_dim, char_hidden_dim, word_hidden_dim, output_dim, max_chars, max_len):
    """Creates a BiLSTM based model

    Arguments:
        input_dim {int} -- Size of the input vocabulary + 1 (adjusted for 0) i.e number of unqique characters
        embed_dim {int} -- Size of character embeddings
        hidden_dim {int} -- Size of LSTM hidden state
        output_dim {int} -- Number of output classes

    Returns:
        keras.models.Sequntial -- Instance of a Keras Sequential model
    """
    dropout_prob = 0.
    r1, r2 = None, None
    # r1=regularizers.l2(0.001)
    # r2=regularizers.l2(0.001)

    model = Sequential()
    # Add character embeddings
    model.add(Embedding(input_dim, embed_dim, embeddings_initializer='uniform'))
    reshape_layer_1 = Lambda(lambda x: K.reshape(x, shape=[-1, max_chars, embed_dim]))
    model.add(reshape_layer_1)
    model.add(Bidirectional(LSTM(units=char_hidden_dim, kernel_regularizer=r1), merge_mode='concat'))
    reshape_layer_2 = Lambda(lambda x: K.reshape(x, shape=[-1, max_len, 2 * char_hidden_dim]))
    model.add(reshape_layer_2)
    model.add(Bidirectional(LSTM(units=word_hidden_dim, kernel_regularizer=r2, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(dropout_prob))
    model.add(CRF(output_dim, sparse_target=True))
    # model.add(CRF(output_dim, sparse_target=False))

    # print model's layerwise summary
    print(model.summary())
    return model


def save_pickle(obj, fpath):
    """Saves an object in pickle format at the given file path

    Arguments:
        obj {object} -- An instance of any pickable object
    """
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(fpath):
    """Saves an object in pickle format from the given file path

    Arguments:
        fpath {str} -- Path of the file where the pickle is stored

    Returns:
        object -- Object stored in the pickle file
    """
    with open(fpath, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_dir', action='store', dest='model_dir', default='langid-data-small/task2/models',
                        help='Location to store the models')
    parser.add_argument('--model_file', action='store', dest='model_file', default='task_3.model',
                        help='Filename for the model')
    parser.add_argument('--vocab_file', action='store', dest='vocab_file', default='vocab.pkl',
                        help='Filename for the vocab')
    parser.add_argument('--labels_file', action='store', dest='labels_file', default='labels.pkl',
                        help='Filename for the labels')
    parser.add_argument('--train_path', action='store', dest='train_path', default='./code_switching/data/train_data.tsv',
                        help='Path to training data file')
    parser.add_argument('--dev_path', action='store', dest='dev_path', default='./code_switching/data/dev_data.tsv',
                        help='Path to test data file')
    parser.add_argument('--test_path', action='store', dest='test_path', default='langid/langid.test',
                        help='Path to test data')
    parser.add_argument('--test_output_path', action='store', dest='test_output_path', default='langid/langid-variants.test_labels',
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
    max_chars = 70
    max_len = 60

    if opt.mode == 'train':
        filepath = Path(opt.train_path)
        train_sequences, train_gold_labels = get_raw_data(filepath)
        print(len(train_sequences), len(train_gold_labels))

        # prepare the vocabulary of characters
        vocab = dict()
        char_counter = Counter()
        for seq in train_sequences:
            for token in seq:
                char_counter.update(token)
        for k, _ in char_counter.most_common(256):
            vocab[k] = len(vocab) + 1
        vocab['unk'] = len(vocab) + 1
        save_pickle(vocab, vocab_file)
        print(f'Vocab size: {len(vocab)}')
        print(max_len, max_chars)

        # create a language to idx map
        label2id = {'other': 0, 'en': 1, 'es': 2}
        save_pickle(label2id, labels_file)
        print(f'Labels:\n{label2id}')

        '''
            vectorize the data into numpy tensors
            X: 2D tensor
            Y: 1D tensor
        '''
        X, Y = get_vectorized_data(train_sequences, train_gold_labels, vocab, label2id, max_len, max_chars)
        print(X.shape)
        print(Y.shape)
        print(f'Training data size: {X.shape[0]}')

        # set hyperparams
        input_dim, output_dim = len(vocab) + 1, len(label2id)
        embed_dim, char_hidden_dim, word_hidden_dim = 100, 100, 100

        # create model
        model = make_model(input_dim, embed_dim, char_hidden_dim, word_hidden_dim, output_dim, max_chars, max_len)

        # use Adam optimizer
        optim = Adam(lr=5e-4, clipnorm=5., clipvalue=1.)

        # compile model
        model.compile(loss=crf_loss, metrics=[crf_viterbi_accuracy], optimizer=optim)
        # model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], optimizer=optim)

        # callbacks for saving and early stopping
        ckpt_clbk = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_clbk = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
        callbacks = [ckpt_clbk, early_clbk]

        epochs = 1
        batch_size = 8
        model.fit(x=X[:1000], y=Y[:1000], batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks)

    elif opt.mode == 'test':
        # get the raw texts for each language
        filepath = Path(opt.dev_path)
        dev_sequences, dev_gold_labels = get_raw_data(filepath)
        print(len(dev_sequences), len(dev_gold_labels))

        # load vocab, labels and model
        vocab = load_pickle(vocab_file)
        print(len(vocab))
        label2id = load_pickle(labels_file)
        custom_objects = {'CRF': CRF,
                          'crf_loss': crf_loss,
                          'crf_viterbi_accuracy': crf_viterbi_accuracy}
        model = load_model(model_file, custom_objects=custom_objects)

        # vectorize the dev data
        X_dev, Y_dev = get_vectorized_data(dev_sequences, dev_gold_labels, vocab, label2id, max_len, max_chars)
        print(X_dev.shape, Y_dev.shape)

        # predict the languages
        y_pred = np.argmax(model.predict(X_dev, batch_size=8), axis=-1)
        print(y_pred.shape)
        Y_dev = np.argmax(Y_dev, axis=-1)
        print(Y_dev.shape)

        print(label2id)
        # print metrics
        print(classification_report(Y_dev, y_pred))
        print(confusion_matrix(Y, y_pred))

    elif opt.mode == 'predict':
        # get test data
        test_data = get_raw_texts(opt.test_path)

        # load model and vocab
        vocab = load_pickle(vocab_file)
        label2id = load_pickle(labels_file)
        id2label = {v: k for k, v in label2id.items()}
        model = load_model(model_file)

        # vectorize test data
        X = get_vectorized_texts(test_data, vocab)

        # predict
        y_pred = np.argmax(model.predict(X), axis=-1)
        print(y_pred.shape)

        # write predictions to output file
        with open(opt.test_output_path, 'w') as file:
            for idx in y_pred:
                file.write(f'{id2label[idx]}\n')

    else:
        print('Incorrect operation mode. Valid values for mode are (train/test/predict)')
