# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:56:29 2021

@author: Fede
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from StarclassML.report import report


def RandomForestclf(dataset, param_list, target_class='Type', n_estimators=400, scoring='accuracy'):
    '''
    This function creates an ensamble of decision trees classifier and tune the hyperparameter thanks to randomsearchCV from sklearn.
    For more information, please visit:
    https://scikit-learn.org/stable/modules/tree.html

    Parameters
    ----------
    min_samples_split : integer
        min number of samples for splitting
    max_depth : integer
        Depth of the tree
    min_samples_leaf : integer
        min number of samples to define a leaf
    criterion : str
        Criterion for splitting the nodes.
    n_estimators : integer
        number of trees that partecipate to the classification problem
    ----------   
    
    param_list should be a dictionary with the ranges of the parameters.
    dataset should be a dataframe from pandas.
    '''
    #controllo che siano passati i giusti tipi di variabili.
    assert type(dataset)==pd.core.frame.DataFrame, 'Your dataset should be a pandas dataframe'

    assert type(param_list)==dict, 'Your param_list should be a dictionary. For more info about parameters, please check the documentation'

    assert(target_class)==str, 'Your target_class should be a str with your target class name'

    ##Separo la target class dal dataset
    attributes = [col for col in df.columns if col != target_class]
    X = dataset[attributes].values
    y = dataset[target_class]

    clf = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1, class_weight=None)
    random_search = RandomizedSearchCV(clf, param_distributions=param_list,
                                       n_iter=100, n_jobs=-1, cv=5,
                                       scoring='accuracy')
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)
    clf = random_search.best_estimator_

    scores = cross_val_score(clf, X, y, cv=5)
    return(clf, scores)


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


    ##Creiamo l'ensemble di alberi e settiamo gli iperparametri
    leaf_list = list(np.arange(1, 100, 2))
    samples_list = list(np.arange(2, 100, 2))
    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': samples_list,
                  'min_samples_leaf': leaf_list,
                  'criterion': ['gini', 'entropy']}
    clf = RandomForestclf(df, param_list=param_list, target_class='Type', n_estimators=400, scoring='accuracy')

    ##Vediamo quali sono gli attributi che più impattano nella classificazione
    attributes = [col for col in df.columns if col != 'Type']
    for col, imp in zip(attributes, clf[0].feature_importances_):
        print(col, imp)

    #crossvalidation, veduamo le performance
    scores = clf[1]
    print('Cross validation Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
