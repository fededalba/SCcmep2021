# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:56:29 2021

@author: Uno
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
import logging

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

'''import os
Non funziona, cerca di risolvere
os.environ['PATH'] += os.pathsep + 'C:\\Users\\Uno\\Desktop\\Esame computing\\SCcmep2021\\StarclassML'
from StarclassML.Decisiontreeclassifier import report'''



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


    ##Definisco il train set e il test set
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)


    ##Creiamo l'ensemble di alberi e vediamo quale ci da la miglior performance
    leaf_list = list(np.arange(1,100,2))
    samples_list = list(np.arange(2,100,2))
    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
              'min_samples_split': samples_list,
              'min_samples_leaf': leaf_list,
              'criterion': ['gini', 'entropy'],
             }
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                             min_samples_split=2, min_samples_leaf=1, class_weight=None)
    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=1000, n_jobs= -1)
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_, n_top=3)



    clf = random_search.best_estimator_
    y_pred = clf.predict(X_test)
    y_pred_tr = clf.predict(X_train)
    print('Train Accuracy %s' % accuracy_score(y_train, y_pred_tr))
    print()
    print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

