import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
import os
import time
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


EPOCHS = 10
MAX_LEN = 50
VOCAB_SIZE = 0
VECTOR_DIM = 100
HIDDEN_DIM = 300
BATCH_SIZE = 16

def getData(path,limit):

    curSent = []
    rawData = []
    normalizedData = []

    for line in open(path):
        tok = line.strip().split('\t')

        if tok == [''] or tok == []:
            rawData.append([x[0] for x in curSent])
            normalizedData.append([x[1] for x in curSent])
            curSent = []

        else:
            if len(tok) == 1:
                tok.append('')
            curSent.append(tok)

    if curSent != []:
        rawData.append([x[0] for x in curSent])
        normalizedData.append([x[1] for x in curSent])

    return rawData[:limit], normalizedData[:limit]

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

        for _ in range(MAX_LEN-len(tokenizedSent)):
            tokenizedSent.append(0)

        tokenizedSent=np.array(tokenizedSent,dtype=float)
        tokenizedData.append(tokenizedSent)

    return np.array(tokenizedData)

def getModel(VOCAB_SIZE):

    model = Sequential()

    model.add(Input(shape=(MAX_LEN,)))
    model.add(Embedding(VOCAB_SIZE, output_dim=VECTOR_DIM, input_length=MAX_LEN, mask_zero=True, trainable=True))
    model.add(LSTM(HIDDEN_DIM))
    model.add(RepeatVector(MAX_LEN))
    model.add(LSTM(HIDDEN_DIM, return_sequences = True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    print(model.summary())
    
    return model

def predictTest(model, xTest, yTest, numToWord):

    for i in range(50):

        ans = model.predict(xTest[i:i+1,:])
        ans = K.argmax(ans,-1)

        raw = []
        normalized = []
        normalizedAns = []
        for j in range(MAX_LEN):
            normalizedAns+=[numToWord[ans.numpy()[0][j]]]

        for j in range(MAX_LEN):
            raw += [numToWord[xTest[i][j]]]
            normalized += [numToWord[yTest[i][j]]]
    
        raw = ' '.join(raw)
        normalized = ' '.join(normalized)
        normalizedAns = ' '.join(normalizedAns)

        print(raw,end='\n')
        print(normalized,end='\n')
        print(normalizedAns,end='\n\n')


def trainModel():

    rawDataTrain, normalizedDataTrain = getData('train.norm',1500)

    rawDataTest, normalizedDataTest = getData('dev.norm',100)

    wordToNum = buildDict(rawDataTrain+normalizedDataTrain)
    numToWord = buildDictInv(wordToNum)
    vocab_size = len(wordToNum)+1
    
    rawDataTrain = tokenize(rawDataTrain, wordToNum)
    normalizedDataTrain = tokenize(normalizedDataTrain, wordToNum)

    rawDataTest = tokenize(rawDataTest, wordToNum)
    normalizedDataTest = tokenize(normalizedDataTest, wordToNum)
    
    xTrain = rawDataTrain
    yTrain = normalizedDataTrain

    xTest  = rawDataTest
    yTest  = normalizedDataTest

    yTrainOneHot = one_hot(yTrain,vocab_size, on_value=1, off_value=0)
    yTestOneHot  = one_hot(yTest,vocab_size, on_value=1, off_value=0)

    model       = getModel(vocab_size)

    history     = model.fit(xTrain, yTrainOneHot, validation_data=(xTest, yTestOneHot), batch_size=BATCH_SIZE, epochs=EPOCHS)

    predictTest(model, xTest, yTest, numToWord)

    return model,history

if __name__ == '__main__':
    model,history = trainModel()