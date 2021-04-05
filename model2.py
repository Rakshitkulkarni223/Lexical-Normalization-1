import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
import os
import time
import json
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict 
import tensorflow.keras.backend as K

from tensorflow import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.optimizers import Adam

TEST = 50
EPOCHS = 30
LIMIT = 1500
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

        n = len(x)

        for i in range(n):
            x[i] = re.sub('@[^ ]+','<username>',x[i])
            x[i] = re.sub('http://[^ ]+','<link>',x[i])

            y[i] = re.sub('@[^ ]+','<username>',y[i])
            y[i] = re.sub('http://[^ ]+','<link>',y[i])

        for _ in range(MAX_LEN-n):
            x.append("")
            y.append("")
        
        X.append(x)
        Y.append(y)

        if limit and len(X) == LIMIT:
            break

    return X, Y

def buildDict(data):

    wordToNum = defaultdict(int)
    num = 1
    for sent in data:
        for word in sent:
            if not wordToNum[word]:
                wordToNum[word] = num
                num+= 1

    return wordToNum

def buildDictInv(wordToNum):

    numToWord = defaultdict(str)

    for key in wordToNum.keys():
        numToWord[wordToNum[key]]=key

    return numToWord

def tokenize(data,wordToNum):

    tokenizedData = []

    for sent in data:

        tokenizedSent = []
        for word in sent:
            tokenizedSent.append(wordToNum[word])

        tokenizedSent=np.array(tokenizedSent,dtype=float)
        tokenizedData.append(tokenizedSent)

    return np.array(tokenizedData)

def getModel(VOCAB_SIZE):

    model = Sequential()

    model.add(Input(shape=(MAX_LEN,)))
    model.add(Embedding(VOCAB_SIZE, output_dim=VECTOR_DIM, input_length=MAX_LEN, mask_zero=True, trainable=True))
    model.add(LSTM(HIDDEN_DIM, return_sequences = True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def trainModel():

    raw, normalized = get_data('data.json', True)

    wordToNum = buildDict(raw + normalized)
    numToWord = buildDictInv(wordToNum)
    vocab_size = len(wordToNum)+1
    
    raw = tokenize(raw, wordToNum)
    normalized = tokenize(normalized, wordToNum)
    
    x = raw
    y = normalized

    y = one_hot(y,vocab_size, on_value=1, off_value=0)

    model       = getModel(vocab_size)

    model.fit(x, y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALID_SPLIT)

    y = model.predict(x[:TEST, : ])

    for i in range(TEST):
        for token in range(MAX_LEN):
            print(numToWord[x[i][token]],end=" ")
        print("",end="\n")
        for token in range(MAX_LEN):
            print(numToWord[np.argmax(y[i][token])],end=" ")
        print("",end="\n\n")

if __name__ == '__main__':
    trainModel()