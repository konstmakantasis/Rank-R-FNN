#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:58:21 2018

@author: konstantinosmakantasis
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing


def create_samples():
    X1 = np.random.rand(50,6,10,14)
    X2 = np.random.rand(50,6,10,14)
    
    return np.concatenate((X1, X2), axis=0)

def PCA_transform(X, n_components=10):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    rpca = PCA(n_components=n_components, whiten=False)
    X_1D_reduced = rpca.fit_transform(X_1D)
    X_reduced = np.reshape(X_1D_reduced, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_reduced, rpca


def PCA_transform_test(X, rpca):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    X_1D_reduced = rpca.transform(X_1D)
    X_reduced = np.reshape(X_1D_reduced, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_reduced


def normalize_fit(X):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    normalizer = preprocessing.Normalizer()
    X_1D_norm = normalizer.fit_transform(X_1D)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm, normalizer


def normalize_transform(X, normalizer):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    X_1D_norm = normalizer.transform(X_1D)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm


def scale_fit(X):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    normalizer = preprocessing.StandardScaler()
    X_1D_norm = normalizer.fit_transform(X_1D)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm, normalizer


def scale_transform(X, normalizer):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    X_1D_norm = normalizer.transform(X_1D)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm
    


if __name__=="__main__":
    X = create_samples()
    X_test = create_samples()
    
    Y, rpca = PCA_transform(X, n_components=10)   
    Y_test = PCA_transform_test(X_test, rpca)
    
    Y, normalizer = normalize_fit(X)
    Y_test = normalize_transform(X_test, normalizer)
    
    Y_S, normalizer = scale_fit(X)
    Y_S_test = scale_transform(X_test, normalizer)
    
