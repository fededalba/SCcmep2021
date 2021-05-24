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



def into_a_string(x):
    ''' Prendo una lista di valori e trasformo ogni elemento in stringa'''
    new_labels = [str(element) for element in x]
    return(new_labels)


if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\Uno\\Documents\\Uni\\Computing methods\\Esame\\Stars.csv')
    ##KNN perde la nozione di distanza più il dataset è esteso. A seconda del
    ##dataset che stiamo studiando selezionerò sempre i due attributi più impattanti secondo il decisiontree classifier.

    ##dataset 1
    for col in df.columns:
        if col != 'A_M' and col != 'R' and col != 'Type':
            del df[col]

    ##Trasformo alcune variabili usando il logaritmo in base 10
    columns = ['R']

    for col in columns:
        X = df[col].values
        Y = np.log10(X)
        df[col] = Y

    scaler = MinMaxScaler()
    P = scaler.fit_transform(df.values)
    #da cambiare se si cambia il dataset
    df['R'] = P[:,0]
    df['A_M'] = P[:,1]


    ##Definisco il train set e il test set
    attributes = [col for col in df.columns if col != 'Type']
    X = df[attributes].values
    y = df['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)



    ##Vediamo qual'è il valore di k ideale per il nostro dataset
    param_list = {'n_neighbors': np.arange(1,100)}
    clf = KNeighborsClassifier(n_neighbors=1, weights = 'distance')
    random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=50)
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_, n_top=3)

    clf = random_search.best_estimator_
    y_pred = clf.predict(X_test)
    y_pred_tr = clf.predict(X_train)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


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
        
        plt.show()