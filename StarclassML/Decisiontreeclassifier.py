# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:53:44 2021

@author: Uno
"""

import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pydotplus
from sklearn import tree
from IPython.display import Image
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score



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

def overfitting_function(X_tr, y_tr, X_te, y_te):
    '''Creo un albero per ogni possibile valore dei parametri, stampo il numero di nodi
        per tutti i punti con lo stesso numero di nodi faccio una media
        del training error e test error'''

    nodi = []
    training_errors = []
    test_errors = []

    for depth in range(2,20):
        for min_sample in range(2,100):
            for min_leaf in range(1,100):
                clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, min_samples_split=min_sample, min_samples_leaf=min_leaf)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_tr = clf.predict(X_train)
                tree = clf.tree_
                nodi.append(tree.node_count)
                training_errors.append(1 - accuracy_score(y_train, y_pred_tr))
                test_errors.append(1 - accuracy_score(y_test, y_pred))

    num_nodi = []
    train_averages = []
    test_averages = []
    ##Questa è da migliorare fede
    for i in range(0,len(nodi)-1):
        element = nodi[i]
        for j in range(0,len(nodi)-1):
            if nodi[j] == element:
                train_averages.append(training_errors[j])
                test_averages.append(test_errors[j])
    



if __name__ == '__main__':

    df = pd.read_csv('C:\\Users\\Uno\\Documents\\Uni\\Computing methods\\Esame\\Stars.csv')

    ##Trasformo la variabile target in una stringa.
    labels = df['Type'].values
    del df['Type']
    df['Type'] = into_a_string(labels)



    ##Mappo i valori categorici in numeri interi.(ci faccio una funzione?)
    label_encoders = dict()
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


    ##Splitto i dati in training e test set
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


    ##Esploro lo spazio dei parametri del mio albero per capire quale è quello più ideale
    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
              }
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
    #randomized search estrae valori casualmente dallo spazio dei parametri
    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=100)
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)

    ##seleziono l'albero migliore e stampo il numero di nodi.
    clf = random_search.best_estimator_
    clf = clf.fit(X_train, y_train)
    clf_tree = clf.tree_
    print(f'Il numero di nodi è {clf_tree.node_count}')

    ##Stampiamo l'albero
    os.environ['PATH'] += os.pathsep + 'C:\\Users\\Uno\\anaconda3\\Library\\bin\\graphviz'
    dot_data = tree.export_graphviz(clf, out_file=None,  
                                feature_names=attributes, 
                                class_names=clf.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())


    ##Analizziamo le performance dell'albero.
    y_pred = clf.predict(X_test)
    y_pred_tr = clf.predict(X_train)
    print('Train Accuracy %s' % accuracy_score(y_train, y_pred_tr))
    print('Train F1-score %s' % f1_score(y_train, y_pred_tr, average=None))
    print()
    print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
    print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


    ##Analizziamo l'overfitting



    ##Vediamo quali sono gli attributi che più impattano nella classificazione
    for col, imp in zip(attributes, clf.feature_importances_):
        print(col, imp)






