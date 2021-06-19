# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:22:46 2021

@author: Uno
"""


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


def onelayer_model():
    '''Costruisco un neural network con un singolo strato'''
    n_feature = X_train.shape[1]
    model = Sequential()
    model.add(Dense(8, input_dim=n_feature, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def twolayers_model():
    '''Costruisco un neural network con un doppio strato'''
    n_feature = X_train.shape[1]
    model = Sequential()
    model.add(Dense(8, input_dim=n_feature, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_DROPOUT_model():
    '''Costruisco una deep neural network con dropout.'''
    model = Sequential()

    n_feature = X_train.shape[1]
    h_dim = 64
    model.add(Dense(h_dim, activation='relu', input_shape=(n_feature,)))
    model.add(Dropout(0.5))
    model.add(Dense(h_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(h_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(h_dim, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    PATH_ACTUAL = os.getcwd()
    PATH = PATH_ACTUAL + "/data/Stars.csv"
    df = pd.read_csv(PATH)


    df['logR'] = np.log10(df['R'].values)
    df = df.filter(items=['logR', 'A_M', 'Temperature', 'Type'])

    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']


    CV_scores1 = []
    CV_scores2 = []
    skf = StratifiedKFold(n_splits=5)
    es = EarlyStopping(monitor='val_loss', patience=50)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model1 = onelayer_model()
        history1 = model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=5, callbacks=[es]).history

        model2 = twolayers_model()
        history2 = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=5, callbacks=[es]).history


        test_loss_1, test_acc_1 = model1.evaluate(X_test, y_test)
        test_loss_2, test_acc_2 = model2.evaluate(X_test, y_test)

        CV_scores1.append(test_acc_1)
        CV_scores2.append(test_acc_2)

    CV_mean1 = np.average(CV_scores1)
    CV_mean2 = np.average(CV_scores2)
    CV_std1 = np.std(CV_scores1)
    CV_std2 = np.std(CV_scores2) 

    #Vediamo cosa otteniamo facendo un modello più complesso con dropout
    CV_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model = build_DROPOUT_model()
        history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=1000, batch_size=5,
                            callbacks=[es]).history


        test_loss, test_acc = model.evaluate(X_test, y_test)

        CV_scores.append(test_acc)

    CV_mean = np.average(CV_scores)
    CV_std = np.std(CV_scores)
    print(f'La cross validation accuracy per il modello dropout è {CV_mean} +- {CV_std}')
    print(f'La cross validation accuracy per il modello 1 è {CV_mean1} +- {CV_std1}')
    print(f'La cross validation accuracy per il modello 2 è {CV_mean2} +- {CV_std2}')
    
        