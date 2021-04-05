import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
import os
import time
import json
import numpy as np

import tensorflow as tf

from tensorflow import keras as K
from tensorflow.keras.models import Sequential

from tensorflow.keras.initializers import Constant

from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import TrueNegatives
from tensorflow.keras.metrics import TruePositives
from tensorflow.keras.metrics import FalsePositives
from tensorflow.keras.metrics import FalseNegatives

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import TimeDistributed

PAD = 0
ONE = 2
ZERO = 1
EPOCHS = 3
LIMIT = 100
MAX_LEN = 50
BATCH_SIZE = 16
HIDDEN_DIM = 100
VECTOR_DIM = 100
VALID_SPLIT = 0.15

def get_data(path, limit=False):

    X = []
    Y = []

    data = json.load(open(path))

    for dict in data:

        x = dict['input']
        y = dict['output']

        for i in range(len(x)):
            x[i] = re.sub('@[^ ]+','<username>',x[i])
            x[i] = re.sub('http://[^ ]+','<link>',x[i])

            y[i] = re.sub('@[^ ]+','<username>',y[i])
            y[i] = re.sub('http://[^ ]+','<link>',y[i])

            if x[i] != y[i]:
                y[i] = 1
            else:
                y[i] = 0
        
        X.append(x)
        Y.append(y)

        if limit and len(X) == LIMIT:
            break

    return X, Y

def tokenize(x):

    tokenized_x = []

    dict = {'<pad>':0}

    num = 1

    for sent in x:

        tokenized_xi = []

        for _ in range(MAX_LEN-len(sent)):
            sent.append('<pad>')

        for word in sent:

            if word not in dict.keys():
                dict[word] = num
                num += 1

            tokenized_xi.append(dict[word])

        tokenized_x.append(np.array(tokenized_xi))

    return np.array(tokenized_x), num

def numpy_format(y):

    np_y = []

    for sent in y:

        for _ in range(MAX_LEN-len(sent)):

            sent.append(0)
        
        np_y.append(np.array(sent).reshape(MAX_LEN,1))

    return np.array(np_y)


def get_model(VOCAB_SIZE):

    model = Sequential()
    model.add(Input(shape=(MAX_LEN)))
    model.add(Embedding(VOCAB_SIZE,
                        output_dim = VECTOR_DIM,
                        input_length = MAX_LEN,
                        mask_zero = True,
                        trainable = True))
    model.add(LSTM(HIDDEN_DIM,return_sequences = True, recurrent_dropout = 0.5, dropout = 0.5))
    model.add(LSTM(1,return_sequences = True, recurrent_dropout = 0.5, dropout = 0.5, activation='sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[Precision(), Recall()],sample_weight_mode="temporal")

    return model

def get_sample_weights(x, y):

    sample_weights = np.zeros((len(y), MAX_LEN))

    sample_weights[y[:,:,0] == 0] = ZERO
    sample_weights[y[:,:,0] == 1] = ONE
    sample_weights[x == 0] = PAD

    return sample_weights

if __name__ == '__main__':

    x, y = get_data('data.json')

    x, VOCAB_SIZE = tokenize(x)
    y = numpy_format(y)

    model = get_model(VOCAB_SIZE)

    print(model.summary())

    sample_weights = get_sample_weights(x, y)

    model.fit(x, y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALID_SPLIT, sample_weight=sample_weights)