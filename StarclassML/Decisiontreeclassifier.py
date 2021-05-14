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

if __name__ == '__main__':

    df = pd.read_csv('C:\\Users\\Uno\\Documents\\Uni\\Computing methods\\Esame\\Stars.csv')

    ##Trasformo la variabile target in una stringa.
    labels = df['Type'].values
    del df['Type']

    df['Type'] = into_a_string(labels)



    ##Mappo i valori categorici in numeri interi.
    from sklearn.preprocessing import LabelEncoder
    label_encoders = dict()
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


