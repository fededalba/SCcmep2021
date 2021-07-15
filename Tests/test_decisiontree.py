# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:05:05 2021

@author: Uno
"""


import unittest
import numpy as np
import pandas as pd
from StarclassML.Decisiontreeclassifier import Decisiontreeclf

class testdecisiontree(unittest.TestCase):
    '''Unit test for Decisiontreeclf'''

    def test_targetclass(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
                  'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
                  'criterion' : ['gini', 'entropy']}
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        df = pd.DataFrame(X)
        y = 5
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, Decisiontreeclf, df, paramlist, y)

    def test_dataset(self):
        '''
        '''

        paramlist = {'max_depth': [None] + list(np.arange(2, 20)),
                  'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
                  'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
                  'criterion' : ['gini', 'entropy']}
        df = 'a wrong value'
        y = 'Type'
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, Decisiontreeclf, df, paramlist, y)

    def test_paramlist(self):
        '''
        '''

        paramlist = 'a wrong value'
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        df = pd.DataFrame(X)
        y = 'Type'
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, Decisiontreeclf, df, paramlist, y)

if __name__ == '__main__':
    unittest.main()
