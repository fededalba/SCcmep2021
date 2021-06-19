# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:18:14 2021

@author: Uno
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
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


    df['logR'] = np.log10(df['R'].values)
    del df['L']
    y = df['Type'].values

    ##Trasformo i valori categorici in numeri interi
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    ##definisco i dataset con cui lavorerò
    df2 = df.filter(items=['logR', 'Temperature'])
    df2R = df.filter(items=['logR', 'A_M'])
    df3 = df.filter(items=['logR', 'Temperature', 'Spectral_Class'])
    df3R = df.filter(items=['logR', 'A_M', 'Temperature'])
    dataarray = [df2, df2R, df3, df3R]

    ##Definiamo il classificatore prima del ciclo for
    param_list = {'n_neighbors': np.arange(1, 100)}
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=50)
    CV_scores = []
    CV_std = []
    labels = ['df2', 'df2R', 'df3', 'df3R', 'df2pca', 'df3pca']
    for dataset in dataarray:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(dataset.values)

        ##hyperparameter tuning
        random_search.fit(X, y)
        report(random_search.cv_results_, n_top=3)

        clf = random_search.best_estimator_

        ##Vediamo le performance usando la crossvalidation
        scores = cross_val_score(clf, X, y, cv=5)
        print('Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        CV_scores.append(scores.mean())
        CV_std.append(scores.std())


    ##Eseguiamo adesso un analisi con PCA
    scaler = StandardScaler()
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    X = scaler.fit_transform(X)
    for i in range(2, 4):
        ##Ad ogni ciclo seleziono una dimensione sempre più grande
        pca = PCA(n_components=i)
        pca.fit(X)
        X_pca = pca.transform(X)

        ##hyperparameter tuning
        random_search.fit(X_pca, y)
        report(random_search.cv_results_, n_top=3)

        clf = random_search.best_estimator_

        ##Vediamo le performance usando la crossvalidation
        scores = cross_val_score(clf, X, y, cv=3)
        print('Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        CV_scores.append(scores.mean())
        CV_std.append(scores.std())

    #Mettiamo in un grafico i risultati ottenuti
    plt.figure()
    sns.set_theme(style='darkgrid')
    plt.ylabel('CV scores')
    #plt.plot(labels, CV_scores, 'o', color = 'black')
    plt.errorbar(labels, CV_scores, CV_std, fmt='.', color='black')
    plt.show()


    ##Adesso vediamo uan rappresentazione del classificatore in 2d
    X = df2R.values
    #Questa volta splittiamo in test set e train set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
    h = 0.2
    #Creo una mappa dei colori
    cmap_light = ListedColormap(['moccasin', 'lightgrey', 'lightcoral', 'pink', 'lightyellow', 'peachpuff'])
    cmap_bold = ListedColormap(['orange', 'gold', 'red', 'crimson', 'darkblue', 'saddlebrown'])
    #Fittiamo i dati
    clf.fit(X_train, y_train)
    #Adesso plottiamo i decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #plottiamo anche i training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    #Cerchiamo i test points
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.ylabel('Absolute magnitude')
    plt.xlabel('logR')
    plt.show()
