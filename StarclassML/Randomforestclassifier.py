# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:56:29 2021

@author: Uno
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score



def report(results, n_top=3):
    '''Questa funzione mi rende gli iperparametri per cui ho ottenuto i migliori top 3 risultati '''
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
    PATH_ACTUAL = os.getcwd()
    PATH = PATH_ACTUAL + "/data/Stars.csv"
    df = pd.read_csv(PATH)

    del df['L']

    ##Trasformiamo la target class in un valore categorico
    stars_type = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf',
                  'Main Sequence', 'Super Giants', 'Hyper Giants']
    df['Type'] = df['Type'].replace(df['Type'].unique(), stars_type)

    ##Trasformo alcune variabili usando il logaritmo in base 10
    df['R'] = np.log10(df['R'].values)

    ##Trasformo le variabili categoriche in numeri interi
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    ##Separo la target class dal dataset
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']


    ##Creiamo l'ensemble di alberi e settiamo gli iperparametri
    leaf_list = list(np.arange(1, 100, 2))
    samples_list = list(np.arange(2, 100, 2))
    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': samples_list,
                  'min_samples_leaf': leaf_list,
                  'criterion': ['gini', 'entropy']}
    clf = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1, class_weight=None)
    random_search = RandomizedSearchCV(clf, param_distributions=param_list,
                                       n_iter=100, n_jobs=-1, cv=5,
                                       scoring='accuracy')
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)
    clf = random_search.best_estimator_

    ##Vediamo quali sono gli attributi che più impattano nella classificazione
    for col, imp in zip(attributes, clf.feature_importances_):
        print(col, imp)

    #crossvalidation, veduamo le performance
    scores = cross_val_score(clf, X, y, cv=5)
    print('Cross validation Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
