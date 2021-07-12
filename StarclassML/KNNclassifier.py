# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:18:14 2021

@author: Fede
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
from StarclassML.report import report


def KNNclf(data, target_class, param_list):
    '''
    The KNN classifier classifies the test point based on their distance to the closest k (k is a parameter) points.
    This function contain also the randomsearchCV for which we can find the best hyperparameter for the classifier.
    Then, it print out the performance of the classifier obtained with crossvalidation.
    For more information, please visit:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Parameters
    ----------
    k : integer
        number of closest points to consider to perform the classification
    ----------

    data has to be the matrix containing the normalized data except the target class.
    Higly suggested perform a data reduction before using knn(ideal dimension is 2 or 3)
    '''
    #controllo che i dati siano un array n dimensionale di numpy.
    assert type(data) == np.ndarray, 'Your data should be a n dimensional numpy array'

    #controllo che la target class sia un array di numpy.
    assert type(target_class) == np.ndarray, 'Your targetclass should be a n dimensional numpy array'

    #controllo che param_list sia un dizionario
    assert type(param_list) == dict, 'Your param_list should be a dictionary. For more info about parameters, please check the documentation'

    ##hyperparameter tuning
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=50)
    random_search.fit(data, target_class)
    report(random_search.cv_results_, n_top=3)

    clf = random_search.best_estimator_
    scores = cross_val_score(clf, data, target_class, cv=5)
    return(clf, scores)

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
    CV_scores = []
    CV_std = []
    labels = ['df2', 'df2R', 'df3', 'df3R', 'df2pca', 'df3pca']
    for dataset in dataarray:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(dataset.values)

        clf = KNNclf(X, y, param_list=param_list)

        ##Vediamo le performance
        scores = clf[1]
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

        clf = KNNclf(X_pca, y, param_list=param_list)

        ##Vediamo le performance
        scores = clf[1]
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
