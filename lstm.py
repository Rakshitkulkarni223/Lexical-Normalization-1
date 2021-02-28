import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict 
import tensorflow.keras.backend as K

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN

EPOCHS = 10
MAX_LEN = 20
VECTOR_DIM = 100
BATCH_SIZE = 128

def getData(path):

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

    return rawData, normalizedData

def buildDict(data):

    wordToNum = defaultdict(int)

    num = 1
    for sent in data:
        for word in sent:
            if not wordToNum[word]:
                wordToNum[word] = num
                num += 1

    return wordToNum

def tokenize(data,wordToNum):

    tokenizedData = []

    for sent in data:

        tokenizedSent = []
        for word in sent:
            tokenizedSent.append(wordToNum[word])
        for i in range(MAX_LEN-len(tokenizedSent)):
            tokenizedSent.append(0)

        tokenizedSent=np.array(tokenizedSent,dtype=float)
        tokenizedData.append(tokenizedSent)

    return np.array(tokenizedData)

def getModel(wordToNum): 

    dModel = VECTOR_DIM
    vocabSize = len(wordToNum)

    model = Sequential()

    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    # PENDING..............
    
    print(model.summary())
    
    return model

def trainModel():

    rawDataTrain, normalizedDataTrain = getData('train.norm')
    rawDataTest, normalizedDataTest = getData('dev.norm')

    wordToNum = buildDict(rawDataTrain+normalizedDataTrain)
    
    rawDataTrain = tokenize(rawDataTrain, wordToNum)
    normalizedDataTrain = tokenize(normalizedDataTrain, wordToNum)

    rawDataTest = tokenize(rawDataTest, wordToNum)
    normalizedDataTest = tokenize(normalizedDataTest, wordToNum)

    model       = getModel(wordToNum)
    
    xTrain      = rawDataTrain
    yTrain      = normalizedDataTrain
    xTest       = rawDataTest
    yTest       = normalizedDataTest
    
    history     = model.fit(xTrain, yTrain, validation_data=(xTest,yTest), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    return model,history

def plot(history):
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy and loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

model,history = trainModel()
plot(history)