# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:25:31 2021

@author: Fede
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


def SVMclf(data, target_class, param_list):
    '''
    SVM is a classifier that create an hyperplane that divide our space in a way that each point that on each side i have different classification labels.
    This function return the SVM with the hyperparameter tuned and the cross validated performance.
    For more information, please visit:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    param_list should be a dictionary with the parameters range.
    data should be a matrix(numpy.ndarray) with the normalized data and without the target class.

    Parameters
    ----------
    C : integer
        it represent how many missclassification we would like to avoid.
    kernel : str
        The kernel that we re gonna use
    gamma : integer
        It tells us how much a point influence the closest point. Higher values it means that more points are influenced.
    ----------

    '''
    #controllo che i dati siano un array n dimensionale di numpy.
    assert type(data) == np.ndarray, 'Your data should be a n dimensional numpy array'

    #controllo che la target class sia un array di numpy.
    assert type(target_class) == np.ndarray, 'Your targetclass should be a n dimensional numpy array'

    #controllo che param_list sia un dizionario
    assert type(param_list) == dict, 'Your param_list should be a dictionary. For more info about parameters, please check the documentation'

    ##Adesso trovo migliori iperparametri e performance con la nested cross-validation
    #definisco la crossvalidation interna
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
    #definisco il modello
    clf = SVC(gamma='auto')
    grid_search = GridSearchCV(clf, param_grid=param_list,
                                       n_jobs=-1, cv=cv_inner,
                                       scoring='accuracy')
    #definisco la crossvalidation esterna
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
    #faccio la nested crossvalidation
    scores = cross_val_score(grid_search, data, target_class, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    return scores

if __name__ == '__main__':
    PATH_ACTUAL = os.getcwd()
    PATH = PATH_ACTUAL + "/data/Stars.csv"
    df = pd.read_csv(PATH)

    ##Trasformo le variabili categoriche in numeri interi
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    ##Utilizzo la trasformazione logaritmica per R
    df['logR'] = np.log10(df['R'].values)

    df2 = df.filter(items=['R', 'Temperature'])
    df2R = df.filter(items=['R', 'A_M'])
    df3 = df.filter(items=['R', 'Temperature', 'Spectral_Class'])
    df3R = df.filter(items=['Temperature', 'R', 'A_M'])
    df4 = df.filter(items=['R', 'Temperature', 'Spectral_Class', 'Color'])
    df4R = df.filter(items=['R', 'A_M', 'L', 'Temperature'])
    df6 = df.filter(items=['R', 'A_M', 'L', 'Temperature', 'Color',
                           'Spectral_Class'])

    dataarray = [df2, df3, df4, df6, df2R, df3R, df4R]
    y = df['Type'].values
    param_list = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto', 0.001, 0.01, 0.1, 1]}
    CV_scores = []
    CV_std = []
    labels = ['dim2', 'dim3', 'dim4', 'dim5']


    for dataset in dataarray:
        ##Per il support vector machine mi conviene utilizzare lo standard scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.values)

        #nestedcrossvalidation
        scores = SVMclf(X, y, param_list=param_list)
        print('Nested Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        CV_scores.append(scores.mean())
        CV_std.append(scores.std())
    CV_scores.append(CV_scores[3])
    CV_std.append(CV_std[3])

    ##PCA
    scaler = StandardScaler()
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    X = scaler.fit_transform(X)
    for i in [2, 3, 4, 5]:
        ##Definisco il train set e il test set
        pca = PCA(n_components=i)
        pca.fit(X)
        X_pca = pca.transform(X)
        scores = SVMclf(X_pca, y, param_list=param_list)
        print('Nested Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        CV_scores.append(scores.mean())
        CV_std.append(scores.std())

    ##Separo i punteggi in base al metodo usato per ridurre la dimensione
    CV_corr = CV_scores[:4]
    CV_corrstd = CV_std[:4]
    CV_RF = CV_scores[4:8]
    CV_RFstd = CV_std[4:8]
    CV_pca = CV_scores[8:]
    CV_pcastd = CV_std[8:]


    plt.figure()
    sns.set_theme(style='darkgrid')
    plt.errorbar(labels, CV_corr, yerr=CV_corrstd, fmt='.', color='blue')
    sns.lineplot(labels, CV_corr, label='feature selected by correlation')
    plt.errorbar(labels, CV_RF, yerr=CV_RFstd, fmt='.', color='orange')
    sns.lineplot(labels, CV_RF, label='feature selected by randomforest')
    plt.errorbar(labels, CV_pca, yerr=CV_pcastd, fmt='.', color='green')
    sns.lineplot(labels, CV_pca, label='feature selected by pca')
    plt.ylabel('Nested Cross Validation Mean Accuracy')
    plt.show()
