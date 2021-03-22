import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict 
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LayerNormalization

from tensorflow.keras.optimizers import Adam


TEST = 50
TRAIN = 2500
EPOCHS = 10
LIMIT = 3000
MAX_LEN = 175
VOCAB_SIZE = 0
VECTOR_DIM = 200
HIDDEN_DIM = 300
BATCH_SIZE = 32

def getData(path, limit):

    rawData = []
    normalizedData = []
    
    data = json.load(open(path))

    for dict in data:

        rawSent = ' '.join(dict['input'])
        rawSent = re.sub('@[^ ]+','<username>',rawSent)
        rawSent = re.sub('http[^ ]+','<link>',rawSent)

        normSent = ' '.join(dict['output'])
        normSent = re.sub('@[^ ]+','<username>',normSent)
        normSent = re.sub('http[^ ]+','<link>',normSent)

        rawData.append(rawSent)
        normalizedData.append(normSent)

    return rawData[:limit], normalizedData[:limit]

def getCharToNum(data):

    charToNum = {}

    num = 1
    for sent in data:
        for letter in sent:
            if letter not in charToNum.keys():
                charToNum[letter] = num
                num += 1

    return charToNum

def getNumToChar(charToNum):

    numToChar = {0:' '}

    for char in charToNum.keys():

        numToChar[charToNum[char]] = char

    return numToChar

def tokenize(data, charToNum, repeat):

    tokenizedData = []

    for sent in data:

        tokenizedSent = []

        if repeat:
            while len(tokenizedSent)<MAX_LEN:
                for letter in sent:
                    tokenizedSent.append(charToNum[letter])
                tokenizedSent.append(0)

            tokenizedSent = tokenizedSent[:MAX_LEN]

        else:
            for letter in sent:
                tokenizedSent.append(charToNum[letter])

            if len(tokenizedSent) < MAX_LEN:
                for _ in range(MAX_LEN-len(tokenizedSent)):
                    tokenizedSent.append(0)
            
            tokenizedSent = tokenizedSent[:MAX_LEN]

        tokenizedData.append(tokenizedSent)
    
    return np.array(tokenizedData)

def getModel(VOCAB_SIZE):

    model = Sequential()

    model.add(Input(shape=(MAX_LEN, )))
    model.add(Embedding(VOCAB_SIZE, output_dim=VECTOR_DIM, input_length=MAX_LEN, mask_zero=False, trainable=True))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(HIDDEN_DIM, return_sequences = True))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':

    x, y = getData('data.json',LIMIT)

    xTrain = x[:TRAIN]
    yTrain = y[:TRAIN]

    xTest = x[TRAIN:]
    yTest = y[TRAIN:]

    charToNum = getCharToNum(xTrain + yTrain + xTest + yTest)
    numToChar = getNumToChar(charToNum)

    xTrain = tokenize(xTrain, charToNum, False)
    yTrain = tokenize(yTrain, charToNum, True)
    xTest = tokenize(xTest, charToNum, False)
    yTest = tokenize(yTest, charToNum, True)

    VOCAB_SIZE = len(charToNum)+1

    yTrain = tf.one_hot(yTrain, VOCAB_SIZE)
    yTest = tf.one_hot(yTest, VOCAB_SIZE)

    model = getModel(VOCAB_SIZE)

    print(model.summary())

    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=EPOCHS, batch_size=BATCH_SIZE)

    for i in range(TEST):

      sent = xTest[i:i+1]
      ans = model.predict(sent)
      ans = K.argmax(ans,-1)

      raw = []
      normalizedAns = []

      for j in range(MAX_LEN):
          normalizedAns+=[numToChar[ans.numpy()[0][j]]]

      for j in range(MAX_LEN):
          raw += [numToChar[xTest[i][j]]]
  
      raw = ''.join(raw)
      normalizedAns = ''.join(normalizedAns)

      print(raw,end='\n')
      print(normalizedAns,end='\n\n')