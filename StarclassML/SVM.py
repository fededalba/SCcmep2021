# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:25:31 2021

@author: Uno
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
    y = df['Type']

    clf = SVC(gamma='auto')
    param_list = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto', 0.001, 0.01, 0.1, 1]}
    grid_search = GridSearchCV(clf, param_grid=param_list,
                               scoring='accuracy', n_jobs=-1)

    CV_scores = []
    CV_std = []
    labels = ['dim2', 'dim3', 'dim4', 'dim5']


    for dataset in dataarray:
        ##Per il support vector machine mi conviene utilizzare lo standard scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.values)

        ##Hyperparameter tuning
        grid_search.fit(X, y)
        report(grid_search.cv_results_, n_top=3)
        clf = grid_search.best_estimator_

        #crossvalidation
        scores = cross_val_score(clf, X, y, cv=5)
        print('Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
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

        ##hyperparameter tuning
        grid_search.fit(X_pca, y)
        print(3)
        report(grid_search.cv_results_, n_top=3)
        clf = grid_search.best_estimator_

        ##Vediamo le performance usando la crossvalidation
        scores = cross_val_score(clf, X, y, cv=5)
        print('Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
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
    plt.ylabel('Cross Validation Mean Accuracy')
    plt.show()
