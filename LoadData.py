#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:38:58 2018

@author: konstantinosmakantasis
"""

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn import preprocessing


def preprocess_data(r_mat, r_dict, gt_mat, gt_dict, window_sz, pca=True, noise_percentage=0.2, noise=False):
    datasets = sio.loadmat(r_mat)
    hypercube = datasets[r_dict]
    
    if noise == True:
        sz1 = hypercube.shape[0]
        sz2 = hypercube.shape[1]
        sz3 = hypercube.shape[2]
        hypercube = np.reshape(hypercube, (-1, sz3))
        hypercube = hypercube + np.random.randn(hypercube.shape[1]) * noise_percentage * np.max(hypercube, axis=0)
        hypercube = np.reshape(hypercube, (sz1, sz2, sz3))

    datasets = sio.loadmat(gt_mat)
    ground_truth = datasets[gt_dict]

    del datasets

    if pca==True:
        n_components = 10
    
        hypercube_1D = np.reshape(hypercube, (-1, hypercube.shape[2]))
        rpca = PCA(n_components=n_components, whiten=True)
        hypercube_1D_reduced = rpca.fit_transform(hypercube_1D)
        hypercube_reduced = np.reshape(hypercube_1D_reduced, (hypercube.shape[0], hypercube.shape[1], -1))
    
        print(rpca.explained_variance_ratio_.sum())
        
        while rpca.explained_variance_ratio_.sum() < 0.999:
            n_components = n_components + 5
            rpca = PCA(n_components=n_components, whiten=True)
            hypercube_1D_reduced = rpca.fit_transform(hypercube_1D)
            hypercube_reduced = np.reshape(hypercube_1D_reduced, (hypercube.shape[0], hypercube.shape[1], -1))
    
            print(rpca.explained_variance_ratio_.sum())
            
            
        print("Final number of components:%d"%n_components)
    
    else:
        n_components = hypercube.shape[2]
        rpca = None
        hypercube_reduced = hypercube #/ np.max(hypercube)
        
    
    window_pad = int(window_sz/2)
    dataset_matrix_size = ((hypercube_reduced.shape[0]-window_pad) * (hypercube_reduced.shape[1]-window_pad), window_sz, window_sz, hypercube_reduced.shape[2])
    dataset_matrix = np.zeros(dataset_matrix_size, dtype=np.float32)
    label_vector = np.zeros((dataset_matrix.shape[0],), dtype=np.int64)

    data_index = 0
    for r in range(hypercube_reduced.shape[0]):
        if r < window_pad or r > hypercube_reduced.shape[0] - window_pad-1:
            continue
        for c in range(hypercube_reduced.shape[1]):
            if c < window_pad or c > hypercube_reduced.shape[1] - window_pad-1:
                continue
        
            patch = hypercube_reduced[r-window_pad:r+window_pad+1, c-window_pad:c+window_pad+1]
            dataset_matrix[data_index,:,:,:] = patch
            label_vector[data_index] = ground_truth[r,c]        
        
            data_index = data_index + 1
        

    dataset_matrix_r = dataset_matrix[label_vector>0,:,:,:]
    label_vector_r = label_vector[label_vector>0]

    rand_perm = np.random.permutation(label_vector_r.shape[0])
    dataset_matrix_r = dataset_matrix_r[rand_perm,:,:,:]
    label_vector_r = label_vector_r[rand_perm]
    
    label_vector_r = label_vector_r - 1.0
    
    return dataset_matrix, label_vector, dataset_matrix_r, label_vector_r, n_components, rpca



def load_data_multi_samples(dataset_matrix_r, label_vector_r, n_components, n_classes, samples):    
    s_sum = 0    
    for i in range(n_classes):
        idx = np.where(label_vector_r == i)   
        if idx[0].shape[0] < samples:
            s_samples = int(idx[0].shape[0]*4 / 5)
        else:
            s_samples = samples
        s_sum = s_sum + s_samples
    
    test_set = np.zeros((dataset_matrix_r.shape[0] - s_sum, dataset_matrix_r.shape[1], dataset_matrix_r.shape[2], dataset_matrix_r.shape[3]), dtype=np.float32)
    l_test_set = np.zeros((dataset_matrix_r.shape[0] - s_sum), dtype=np.int64)
    train_set = np.zeros((s_sum, dataset_matrix_r.shape[1], dataset_matrix_r.shape[2], dataset_matrix_r.shape[3]), dtype=np.float32)    
    l_train_set = np.zeros((s_sum), dtype=np.int64)
    
    count_start = 0
    count_train = 0
    for i in range(n_classes):
        idx = np.where(label_vector_r == i)
        rand_perm = np.random.permutation(idx[0].shape[0])
        class_i = dataset_matrix_r[idx]
        class_i = class_i[rand_perm]

        if idx[0].shape[0] < samples:
            s_samples = int(idx[0].shape[0]*4 / 5)
        else:
            s_samples = samples        
        
        count_end = count_start + idx[0].shape[0] - s_samples    
        
        train_set[count_train:count_train+s_samples,:] = class_i[0:s_samples]
        l_train_set[count_train:count_train+s_samples] = i
        count_train = count_train+s_samples
        
        test_set[count_start:count_end,:] = class_i[s_samples:]
        l_test_set[count_start:count_end] = i
        
        count_start = count_end
        
        
    rand_perm = np.random.permutation(l_train_set.shape[0])
    l_train_set = l_train_set[rand_perm]
    train_set = train_set[rand_perm]
    
    rand_perm = np.random.permutation(l_test_set.shape[0])
    l_test_set = l_test_set[rand_perm]
    test_set = test_set[rand_perm]
         
    test_set_x, test_set_y = (test_set, l_test_set)
    valid_set_x, valid_set_y = (test_set, l_test_set)
    train_set_x, train_set_y = (train_set, l_train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval


def batch_creation(x, y, batch_size=100):
    total_size = x.shape[0]
    
    batches = []
    num_batches = int(total_size/batch_size)
    for i in range(num_batches):
        batches.append([x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]])
        
    batches.append([x[(i+1)*batch_size:], y[(i+1)*batch_size:]])
        
        
    return batches


def scale_fit(X):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    normalizer = preprocessing.StandardScaler()
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_1D_norm = normalizer.fit_transform(X_1D)
    X_1D_norm = minmax.fit_transform(X_1D_norm)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm, normalizer, minmax


def scale_transform(X, normalizer, minmax):
    X_1D = np.reshape(X, (-1, X.shape[3]))
    X_1D_norm = normalizer.transform(X_1D)
    X_1D_norm = minmax.transform(X_1D_norm)
    X_norm = np.reshape(X_1D_norm, (X.shape[0], X.shape[1], X.shape[2], -1))
    
    return X_norm

    
    
    
    
    
    
    
    
    
    
    
    
    