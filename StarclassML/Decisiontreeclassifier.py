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
from sklearn.model_selection import KFold

def Decisiontreeclf(dataset, param_list, target_class='Type', n_iter=100):
    '''
    This function creates the decision tree classifier with the hyperparameter already tuned.
    The decision tree works that way:He select an attribute and build a node where data can
    be split depending on an objective function (entropy, gini). So at each step(node) the
    classifier make a decision. For more information, please visit:
    https://scikit-learn.org/stable/modules/tree.html
    param_list should be a dictionary with the ranges of the parameters.
    dataset should be a dataframe from pandas.

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

    '''
    #controllo che siano passati i giusti tipi di variabili.
    assert type(dataset)==pd.core.frame.DataFrame, 'Your dataset should be a pandas dataframe'

    assert type(param_list)==dict, 'Your param_list should be a dictionary. For more info about parameters, please check the documentation'

    assert type(target_class)==str, 'Your target_class should be a str with your target class name'

    ##Separo la target class dal dataset
    attributes = [col for col in df.columns if col != target_class]
    X = dataset[attributes].values
    y = dataset[target_class]

    ##Adesso trovo migliori iperparametri e performance con la nested cross-validation
    #definisco la crossvalidation interna
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
    #definisco il modello
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1)
    random_search = RandomizedSearchCV(clf, param_distributions=param_list,
                                       n_iter=n_iter, n_jobs=-1, cv=cv_inner,
                                       scoring='accuracy')
    #definisco la crossvalidation esterna
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
    #faccio la nested crossvalidation
    scores = cross_val_score(random_search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    return scores


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

    scores = Decisiontreeclf(df, param_list=param_list, target_class='Type', n_iter=100)

    ''''##Stampiamo l'albero
    os.environ['PATH'] += os.pathsep + 'C:\\Users\\Uno\\anaconda3\\Library\\bin\\graphviz'
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=attributes,
                                    class_names=clf.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())'''

    #stampiamo i risultati
    print('Nested Cross validation Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
