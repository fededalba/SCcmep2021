# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:04:42 2021

@author: feder
"""
import os
import numpy as np
import pandas as pd
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
from IPython.display import Image


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

    ##Trasformiamo la target class in un valore categorico
    stars_type = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence',
                  'Super Giants', 'Hyper Giants']
    df['Type'] = df['Type'].replace(df['Type'].unique(), stars_type)
    del df['L']



    ##Mappo i valori categorici in numeri interi.
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    ##Trasformo alcune variabili usando il logaritmo in base 10
    df['R'] = np.log10(df['R'].values)


    ##Separo la target class dal dataset
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']

    ##hyperparameter tuning
    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
                  'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
                  'criterion' : ['gini', 'entropy']}
    clf = DecisionTreeClassifier(
        criterion='gini', max_depth=None, min_samples_split=2,
        min_samples_leaf=1)
    #randomized search estrae valori casualmente dallo spazio dei parametri
    random_search = RandomizedSearchCV(clf, param_distributions=param_list,
                                       n_iter=200, scoring='accuracy')
    random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)

    ##seleziono l'albero migliore e stampo il numero di nodi.
    clf = random_search.best_estimator_
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


    #crossvalidation
    scores = cross_val_score(clf, X, y, cv=5)
    print('Cross validation Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
