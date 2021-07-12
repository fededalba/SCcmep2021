# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 16:44:15 2021

@author: Uno
"""


import unittest
import numpy as np
from StarclassML.SVM import SVMclf

class testsvm(unittest.TestCase):
    '''Unit test for svmclf'''

    def test_targetclass(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto', 0.001, 0.01, 0.1, 1]}
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        y = 'a wrong value'
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, SVMclf, X, y, paramlist)

    def test_data(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto', 0.001, 0.01, 0.1, 1]}
        y = np.array([1,2,3,4,5])
        X = 'a wrong value'
        #Se nella funzione esce un assertion error, allora il test hka successo.
        self.assertRaises(AssertionError, SVMclf, X, y, paramlist)

    def test_paramlist(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = 'another wrong value'
        y = np.array([1,2,3,4,5])
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, SVMclf, X, y, paramlist)

if __name__ == '__main__':
    unittest.main()
