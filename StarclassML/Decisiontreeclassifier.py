# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:04:42 2021

@author: feder
"""
import os
import sys
import numpy as np
import pandas as pd
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
from IPython.display import Image
'''
PATH_ACTUAL = os.getcwd()
PATH_PARENT = os.path.abspath(os.path.join(PATH_ACTUAL, os.pardir))
#sys.path.append(PATH_PARENT)'''

from StarclassML.report import report

def Decisiontreeclf(dataset, param_list, target_class='Type', n_iter=100, scoring='accuracy'):
    '''
    This function creates the decision tree classifier with the hyperparameter already tuned.
    The decision tree works that way:He select an attribute and build a node where data can
    be split depending on an objective function (entropy, gini). So at each step(node) the
    classifier make a decision. For more information, please visit:
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
    ----------   
    
    param_list should be a dictionary with the ranges of the parameters.
    dataset should be a dataframe from pandas.
    '''
    #controllo che siano passati i giusti tipi di variabili.
    assert type(dataset)==pd.core.frame.DataFrame

    assert type(param_list)==dict

    assert(target_class)==str

    ##Separo la target class dal dataset
    attributes = [col for col in dataset.columns if col != target_class]
    X = dataset[attributes].values
    y = dataset[target_class]

    ##hyperparameter tuning
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

    #crossvalidation
    scores = cross_val_score(clf, X, y, cv=5)
    return(clf, scores)


if __name__ == '__main__':

    PATH_ACTUAL = os.getcwd()
    PATH_DATA = PATH_ACTUAL + "/data/Stars.csv"
    df = pd.read_csv(PATH_DATA)

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

    param_list = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
                  'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
                  'criterion' : ['gini', 'entropy']}

    clf = Decisiontreeclf(df, param_list=param_list, target_class='Type', n_iter=100, scoring='accuracy')

    ''''##Stampiamo l'albero
    os.environ['PATH'] += os.pathsep + 'C:\\Users\\Uno\\anaconda3\\Library\\bin\\graphviz'
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=attributes,
                                    class_names=clf.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())'''


    #crossvalidation
    scores = clf[1]
    print('Cross validation Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
