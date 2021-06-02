# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:25:31 2021

@author: Uno
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
#from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def into_a_string(x):
    ''' Prendo una lista di valori e trasformo ogni elemento in stringa'''
    new_labels = [str(element) for element in x]
    return(new_labels)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\Uno\\Documents\\Uni\\Computing methods\\Esame\\Stars.csv')

    ##Trasformo la variabile target in una stringa.
    labels = df['Type'].values
    del df['Type']
    df['Type'] = into_a_string(labels)

    ##Trasformo le variabili categoriche in numeri interi
    label_encoders = dict()
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    ##Utilizzo la trasformazione logaritmica per R
    columns = ['L','R']
    for col in columns:
        X = df[col].values
        Y = np.log10(X)
        df[col] = Y


    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']

    ##Per il support vector machine mi conviene utilizzare lo standard scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100, stratify=y)

    ##Hyperparameter tuning
    clf = SVC(gamma='auto')
    param_list = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear','rbf', 'poly'], 'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1,1, 100]}
    grid_search = GridSearchCV(clf, param_grid=param_list, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    report(grid_search.cv_results_, n_top=3)
    clf = grid_search.best_estimator_

    ##Vediamo le performance del nostro classificatore
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy %s' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
