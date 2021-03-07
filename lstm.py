import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict 
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.optimizers import Adam


EPOCHS = 15
MAX_LEN = 50
VOCAB_SIZE = 0
VECTOR_DIM = 80
HIDDEN_DIM = 100
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

def getModel(VOCAB_SIZE):

    encoder_inputs = Input(shape=(MAX_LEN,), dtype='int32')
    encoder_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=VECTOR_DIM,input_length=MAX_LEN,trainable=True)(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN,), dtype='int32')
    decoder_embedding =Embedding(input_dim=VOCAB_SIZE, output_dim=VECTOR_DIM,input_length=MAX_LEN,trainable=True)(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    outputs = TimeDistributed(Dense(VOCAB_SIZE,activation='softmax'))(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)

    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy')
    
    print(model.summary())
    
    return model


def trainModel():

    rawDataTrain, normalizedDataTrain = getData('train.norm',800)
    rawDataTest, normalizedDataTest = getData('dev.norm',200)

    wordToNum = buildDict(rawDataTrain+normalizedDataTrain)
    vocab_size = len(wordToNum)+1
    
    rawDataTrain = tokenize(rawDataTrain, wordToNum)
    normalizedDataTrain = tokenize(normalizedDataTrain, wordToNum)

    rawDataTest = tokenize(rawDataTest, wordToNum)
    normalizedDataTest = tokenize(normalizedDataTest, wordToNum)
    
    xTrain = rawDataTrain
    yTrain = normalizedDataTrain

    xTest  = rawDataTest
    yTest  = normalizedDataTest

    yTrainOneHot = tf.one_hot(yTrain,vocab_size)
    yTestOneHot  = tf.one_hot(yTest,vocab_size)

    model       = getModel(vocab_size)

    history     = model.fit([xTrain, yTrain], yTrainOneHot, validation_data=([xTest, yTest], yTestOneHot), batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    return model,history


if __name__ == "__main__":
    model,history = trainModel()