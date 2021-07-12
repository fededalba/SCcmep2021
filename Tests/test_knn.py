# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:26:23 2021

@author: Uno
"""


import unittest
import numpy as np
from StarclassML.KNNclassifier import KNNclf

class testsvm(unittest.TestCase):
    '''Unit test for KNNclf'''

    def test_targetclass(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = {'n_neighbors': np.arange(1, 100)}
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        y = 'a wrong value'
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, KNNclf, X, y, paramlist)

    def test_data(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = {'n_neighbors': np.arange(1, 100)}
        y = np.array([1,2,3,4,5])
        X = 'a wrong value'
        #Se nella funzione esce un assertion error, allora il test hka successo.
        self.assertRaises(AssertionError, KNNclf, X, y, paramlist)

    def test_paramlist(self):
        '''Testo che la funzione mi renda un assertionerror in caso riceva variabili in una forma diversa
        da quella richiesta'''
        paramlist = 'another wrong value'
        y = np.array([1,2,3,4,5])
        X = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        #Se nella funzione esce un assertion error, allora il test ha successo.
        self.assertRaises(AssertionError, KNNclf, X, y, paramlist)

if __name__ == '__main__':
    unittest.main()