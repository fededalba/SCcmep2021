# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:44:09 2021

@author: Uno
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    PATH_ACTUAL = os.getcwd()
    PATH = PATH_ACTUAL + "/data/Stars.csv"
    df = pd.read_csv(PATH)

    ##Trasformiamo la target class in un valore categorico
    stars_type = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf',
                  'Main Sequence', 'Super Giants', 'Hyper Giants']
    df['Type'] = df['Type'].replace(df['Type'].unique(), stars_type)


    ##Vediamo come sono distribuite le target class del nostro dataset
    counts = []
    for classes in stars_type:
        counts.append(len(df[df['Type']==classes].values)*100/len(df['Type'].values))
    plt.figure(1)
    plt.pie(counts, labels=stars_type)


    ##Plottiamo le distribuzioni delle variabili
    columns = ['Temperature', 'L', 'R', 'A_M']
    for column in columns:
        plt.figure()
        plt.xlabel(f"{column}")
        sns.histplot(df, x=df[column].values, fill=False)
        sns.histplot(df, x=df[column].values, hue='Type', element='step')
        plt.show()

    ##Trasformo la variabile raggio.
    df['R'] = np.log10(df['R'].values)


    ##Calcolo la matrice di correlazione, però prima trasformiamo i valori categorici in interi
    column2encode = ['Spectral_Class', 'Color']
    for col in column2encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        plt.figure()
        sns.heatmap(df.corr(), annot=True, square=True, mask=mask)

    ##Qualche scatterplot
    plt.figure()
    plt.grid()
    plt.xlabel('Temperature')
    plt.ylabel('A_M')
    sns.scatterplot(df['Temperature'].values, df['A_M'].values, hue=df['Type'])

    plt.figure()
    plt.grid()
    plt.xlabel('R')
    plt.ylabel('A_M')
    sns.scatterplot(df['R'].values, df['A_M'].values, hue=df['Type'])

    plt.figure()
    plt.grid()
    plt.xlabel('A_M')
    plt.ylabel('L')
    sns.scatterplot(df['A_M'].values, np.log10(df['L'].values), hue=df['Type'])
