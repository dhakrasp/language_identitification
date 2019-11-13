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
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          Input, Lambda, Masking, Multiply)
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# set seeds for consistent results
np.random.seed(1)
tf.set_random_seed(2)


def get_raw_data(fpath):
    """Fetches the data from a given TSV data file

    Arguments:
        fpath {str} -- Path to the TSV file holding the data

    Returns:
        {list} -- List of all the samples. A list of list of strings. [['hello', 'world'], ['python', 'rocks']]
        {list} -- List of all the samples' labels. A list of list of strings. [['en', 'es'], ['es', 'other']]
    """
    sequences = []
    gold_labels = []
    with open(fpath, 'r', encoding='utf-8') as file:
        tweet_id = None
        seq = []
        labs = []
        for row in csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE):
            # condition when s new tweet starts
            if int(row[0]) != tweet_id:
                tweet_id = int(row[0])
                # store the sample and labels if sample isn't empy
                if seq:
                    sequences.append(seq)
                    gold_labels.append(labs)
                # clear sample and labels for processing new tweet
                seq = []
                labs = []
            # append token to sample (from 2nd last column)
            seq.append(row[-2])
            # append label to gold labels (from last column)
            labs.append(row[-1])
    return sequences, gold_labels


def get_vectorized_data(sequences, gold_labels, vocab, label2id, max_len, max_chars):
    """Converts sequences and gold label sequences into tensors of integers

    Arguments:
        sequences {list} -- List of all the samples. A list of list of strings. [['hello', 'world'], ['python', 'rocks']]
        gold_labels {list} -- List of all the samples' labels. A list of list of strings. [['en', 'es'], ['es', 'other']]
        vocab {dict} -- A dict holding the word: idx mapping
        label2id {dict} -- A dict holding the language: idx mapping
        max_len {int} -- Max chars in a token
        max_chars {int} -- Max tokens in a sequence

    Returns:
        np.array -- Rank 2 tensor holding all the vectorized training samples. Shape will be (len(sequences), max_chars * max_len)
        np.array -- Rank 3 tensor holding all the vectorized training labels
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
        one_hots = to_categorical(vec, num_classes=4)
        Y.append(one_hots)
        # Y.append(vec)
    Y = np.array(Y)
    return np.reshape(X, newshape=(len(sequences), -1)), Y


def get_vectorized_sequences(sequences, vocab, max_len, max_chars):
    """Converts text data into tensors of integers

    Arguments:
        texts {list} -- List of all the test samples. A list of list of strings [['hello', 'world'], ['python', 'rocks']]
        vocab {dict} -- A dict holding the word: idx mapping
        max_len {int} -- Max chars in a token
        max_chars {int} -- Max tokens in a sequence

    Returns:
        np.array -- Rank 2 tensor holding all the vectorized samples. shape will be (len(sequences), max_chars * max_len)
    """
    X = []
    for seq in sequences:
        for word in seq[:max_len]:
            char_seq = [vocab[c] if c in vocab else vocab['unk'] for c in word[:max_chars]]
            X.append(char_seq)
        for _ in range(max_len - len(seq)):
            X.append([0] * max_chars)
    X = pad_sequences(X, maxlen=max_chars, dtype='int32', padding='post', truncating='post', value=0)
    return np.reshape(X, newshape=(len(sequences), -1))


def make_model(input_dim, embed_dim, char_hidden_dim, word_hidden_dim, output_dim, max_chars, max_len):
    """Creates a 2 level BiLSTM based model

    Arguments:
        input_dim {int} -- Size of the input vocabulary + 1 (adjusted for 0) i.e number of unqique characters
        embed_dim {int} -- Size of character embeddings
        char_hidden_dim {int} -- Size of char-level LSTM hidden state
        word_hidden_dim {int} -- Size of word-level LSTM hidden state
        output_dim {int} -- Number of output classes (4: other, en, es, padding)
        max_chars {int} -- Max chars in a token
        max_len {int} -- Max tokens in a tweet

    Returns:
        keras.models.Model -- Instance of a Keras Model
    """
    dropout_prob = 0.
    r1, r2 = None, None
    # r1=regularizers.l2(0.001)
    # r2=regularizers.l2(0.001)

    inputs = Input(shape=(max_chars * max_len,))
    word_mask = Input(shape=(max_len, 2 * char_hidden_dim))

    '''
        Add character embeddings. Keep mask_zero = True.
        This is need because we have added padding at char level and want to ignore this padding for furture processing.
    '''
    embeddings = Embedding(input_dim, embed_dim, embeddings_initializer='uniform', mask_zero=True)
    embed = embeddings(inputs)

    '''
        Reshape to a rank 3 tensor of (batch_size, max_chars, embed_dim)
        This tensor will hold the representation for each char in each word in each tweet (plus padding as required)
    '''
    reshape_layer_1 = Lambda(lambda x: K.reshape(x, shape=[-1, max_chars, embed_dim]))
    reshaped = reshape_layer_1(embed)

    # Apply char-level LSTM
    char_lstm = Bidirectional(LSTM(units=char_hidden_dim, kernel_regularizer=r1), merge_mode='concat')
    char_hidden = char_lstm(reshaped)

    '''
        Reshape to a rank 3 tensor of (batch_size, max_len, 2 * char_hidden_dim])
        This tensor will essentially capture the representation for each word in each tweet (plus padding as required)
    '''
    reshape_layer_2 = Lambda(lambda x: K.reshape(x, shape=[-1, max_len, 2 * char_hidden_dim]))
    reshaped_2 = reshape_layer_2(char_hidden)

    '''
        Apply the mask at word level. 
        This is need because we have added padding at word level and don't want to compute the loss or metrics for this padding.
    '''
    mask_multiply = Multiply()
    masked = mask_multiply([reshaped_2, word_mask])
    mask_layer = Masking(mask_value=0.)
    masked = mask_layer(masked)

    # Apply the word-level LSTM
    word_lstm = Bidirectional(LSTM(units=word_hidden_dim, kernel_regularizer=r2, return_sequences=True), merge_mode='concat')
    word_hidden = word_lstm(masked)
    dropout = Dropout(dropout_prob)
    dropped = dropout(word_hidden)

    # Project the word-level hidden representation to output space
    dense = Dense(output_dim, activation='softmax')
    output_probs = dense(dropped)

    # create model
    model = Model(inputs=[inputs, word_mask], outputs=output_probs)
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
    parser.add_argument('--model_dir', action='store', dest='model_dir', default='langid-data-small/task3/models',
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

    # assume max chars in a token
    max_chars = 50
    # assume max tokens in a tweet
    max_len = 60

    # dimensions for embedding layer, char-level lstm, word-level lstm
    embed_dim, char_hidden_dim, word_hidden_dim = 100, 100, 100

    if opt.mode == 'train':
        # read the training data
        filepath = Path(opt.train_path)
        train_sequences, train_gold_labels = get_raw_data(filepath)
        assert len(train_sequences) == len(train_gold_labels)
        print(f'No. of training samples: {len(train_sequences)}')

        # prepare and store the vocabulary of characters
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

        # create a language to idx map
        label2id = {'pad': 0, 'other': 1, 'en': 2, 'es': 3}
        save_pickle(label2id, labels_file)
        print(f'Labels:\n{label2id}')

        '''
            vectorize the data into numpy tensors
            X: Rank 2 tensor (#samples, max_chars * max_len)
            Y: Rank 3 tensor
        '''
        # vectorize the dev data
        X, Y = get_vectorized_data(train_sequences, train_gold_labels, vocab, label2id, max_len, max_chars)

        # compute the mask
        word_mask = np.zeros((len(X), max_len, 2 * char_hidden_dim))
        lengths = [min(len(seq), max_len) for seq in train_sequences]
        for i, len_ in enumerate(lengths):
            for j in range(len_):
                word_mask[i, j, :] = 1
        print(X.shape, Y.shape)
        print(f'Training data size: {X.shape[0]}')

        # set hyperparams
        input_dim, output_dim = len(vocab) + 1, len(label2id)

        # create model
        model = make_model(input_dim, embed_dim, char_hidden_dim, word_hidden_dim, output_dim, max_chars, max_len)

        # use Adam optimizer
        optim = Adam(lr=5e-4, clipnorm=5., clipvalue=1.)

        # compile model
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optim)

        # callbacks for saving and early stopping
        ckpt_clbk = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_clbk = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
        callbacks = [ckpt_clbk, early_clbk]

        epochs = 20
        batch_size = 8
        model.fit(x=[X, word_mask], y=Y, batch_size=batch_size, epochs=epochs, validation_split=0.15, callbacks=callbacks)

    elif opt.mode == 'test':
        # get the dev data (tweets and gold labels)
        filepath = Path(opt.dev_path)
        dev_sequences, dev_gold_labels = get_raw_data(filepath)
        print(len(dev_sequences), len(dev_gold_labels))

        # load vocab, labels and model
        vocab = load_pickle(vocab_file)
        label2id = load_pickle(labels_file)
        id2label = {v: k for k, v in label2id.items()}
        model = load_model(model_file)

        # vectorize the dev data
        X_dev, Y_dev = get_vectorized_data(dev_sequences, dev_gold_labels, vocab, label2id, max_len, max_chars)

        # compute the mask
        word_mask = np.zeros((len(X_dev), max_len, 2 * char_hidden_dim))
        lengths = [min(len(seq), max_len) for seq in dev_sequences]
        for i, len_ in enumerate(lengths):
            for j in range(len_):
                word_mask[i, j, :] = 1

        # predict the languages
        y_pred = np.argmax(model.predict([X_dev, word_mask], batch_size=8), axis=-1)

        # flatten the predictions to compute metrics
        y_pred_flat = []
        y_dev_flat = []
        for pred_seq, seq in zip(y_pred, dev_gold_labels):
            pred_seq = list(pred_seq[:len(seq)])
            y_pred_flat += pred_seq
            y_dev_flat += seq[:max_len]
        y_pred_flat = list(map(lambda x: id2label[x], y_pred_flat))

        # compute metrics and confusion matrix
        print(classification_report(y_dev_flat, y_pred_flat))
        print(confusion_matrix(y_dev_flat, y_pred_flat))

    elif opt.mode == 'predict':
        # load vocab, labels, model
        vocab = load_pickle(vocab_file)
        label2id = load_pickle(labels_file)
        id2label = {v: k for k, v in label2id.items()}
        model = load_model(model_file)
        while True:
            # read a tweet from console (as a batch)
            sequences = [input('Type your tweet:\n').split()]

            # vectorize the tweet
            X = get_vectorized_sequences(sequences, vocab, max_len, max_chars)

            # compute word-level mask
            word_mask = np.zeros((len(X), max_len, 2 * char_hidden_dim))
            lengths = [min(len(seq), max_len) for seq in sequences]
            for i, len_ in enumerate(lengths):
                for j in range(len_):
                    word_mask[i, j, :] = 1

            # predict
            y_pred = np.argmax(model.predict([X, word_mask], batch_size=8), axis=-1)
            y_pred_flat = y_pred[0][:len(sequences[0])]

            # convert from numbers to actual labels
            y_pred_flat = list(map(lambda x: id2label[x], y_pred_flat))
            print('')

            # print token-wise result
            for tok, lab in zip(sequences[0], y_pred_flat):
                print(tok, '---->', lab)
            print('-' * 80)

    else:
        print('Incorrect operation mode. Valid values for mode are (train/test/predict)')
