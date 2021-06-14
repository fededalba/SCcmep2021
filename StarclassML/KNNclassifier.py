# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:18:14 2021

@author: Uno
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


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



if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\feder\Downloads\archive\Stars.csv')

    df['logR'] = np.log10(df['R'].values)
    y = df['Type'].values

    ##Trasformo i valori categorici in numeri interi
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    ##definisco i dataset con cui lavorerò
    df2 = df.filter(items= ['R', 'Temperature'])
    df2R = df.filter(items= ['R', 'A_M'])
    df3 = df.filter(items = ['R', 'Temperature', 'Spectral_Class'])
    df3R = df.filter(items = ['R', 'A_M', 'L'])

    dataarray = [df2, df2R, df3, df3R]
    ##Definiamo il classificatore prima del ciclo for
    param_list = {'n_neighbors': np.arange(1,100)}
    clf = KNeighborsClassifier(n_neighbors=1, weights = 'distance')
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


    ##PCA
    scaler = StandardScaler()
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    X = scaler.fit_transform(X)
    for i in range(2,4):
        ##Definisco il train set e il test set
        pca = PCA(n_components= i)
        pca.fit(X)
        X_pca = pca.transform(X)

        ##hyperparameter tuning
        random_search.fit(X_pca, y)
        report(random_search.cv_results_, n_top=3)

        clf = random_search.best_estimator_

        ##Vediamo le performance usando la crossvalidation
        scores = cross_val_score(clf, X, y, cv=5)
        print('Cross validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        CV_scores.append(scores.mean())
        CV_std.append(scores.std())


    plt.figure()
    plt.grid()
    plt.ylabel('CV scores')
    plt.plot(labels, CV_scores, 'o', color = 'black')
    #plt.errorbar(labels, CV_scores, CV_std, fmt = '.', color = 'black')
    plt.show()        

'''
    ##plottiamo
    n_neigh = clf.n_neighbors
    X = X_train
    y = y_train
    h = 0.2
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'green', 'yellow'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue', 'red', 'blue', 'darkgreen'])
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neigh, weights=weights)
        clf.fit(X, y)
        df['Type'] = P[:,2]
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neigh, weights))
        
        plt.show()'''