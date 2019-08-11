#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:49:35 2018

@author: konstantinosmakantasis
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
from sklearn.metrics import accuracy_score
import copy

import LoadData
import PCA


class Net_modes(nn.Module):
    
    def __init__(self, W, V, b, n_classes, hidden):
        super(Net_modes, self).__init__()
        
        with torch.no_grad():
            self.W = copy.deepcopy(W)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True) #initialize these weights
        with torch.no_grad():
            self.fc2.weight = torch.nn.Parameter(copy.deepcopy(V))
            self.fc2.bias = torch.nn.Parameter(copy.deepcopy(b))
        
    def forward(self, x):
        self.W.requires_grad = True
        x = torch.matmul(self.W, x)
        x = torch.diagonal(x, dim1=2, dim2=3)
        x = torch.sum(x, 2)
        x = torch.tanh(x)
        x = self.fc2(x)
        
        return x


class Net_hidden(nn.Module):
    
    def __init__(self, V, b, n_classes, hidden):
        super(Net_hidden, self).__init__()
        
        self.fc1 = nn.Linear(hidden, n_classes, bias=True) #initialize these weights
        with torch.no_grad():
            self.fc1.weight = torch.nn.Parameter(copy.deepcopy(V))
            self.fc1.bias = torch.nn.Parameter(copy.deepcopy(b))
        
    def forward(self, x):
        x = self.fc1(x)
        
        return x
    
    
def set_tensor_matricization(x, mode=0):
    matricization = []
    
    for sample in x:
        matricization.append(tl.unfold(sample, mode))
        
    return np.asarray(matricization)


def set_tensor_vectorization(x):
    vectorization = []
    
    for sample in x:
        vectorization.append(tl.tensor_to_vec(sample))
        
    return np.asarray(vectorization)


def compute_XK_torch(X, K):    
    X_khatri_tensor = torch.zeros(X.size()[0], K.size()[0], X.size()[1], K.size()[2])
    
    for i in range(X.size()[0]):
        X_khatri_tensor[i] = torch.matmul(X[i], K)
        
    return X_khatri_tensor


def compute_khatri_rao(B1, B2):
    H = B1.shape[0]
    khatri = [tl.tenalg.khatri_rao([B1[i].transpose(), B2[i].transpose()]) for i in range(H)]
    
    return np.asarray(khatri)


def compute_khatri_rao_3(B1, B2, B3):
    H = B1.shape[0]
    khatri = [tl.tenalg.khatri_rao([B1[i].transpose(), B2[i].transpose(), B3[i].transpose()]) for i in range(H)]
    
    return np.asarray(khatri)


if __name__ == "__main__":
#########################################################################################################    
# Botswana --> 14 classes plus background                                                               #
#    --> '../multi_data/MultispectralDatasets/Botswana.mat', 'Botswana'                                 #
#    --> '../multi_data/MultispectralDatasets/Botswana_gt.mat', 'Botswana_gt'                           #
#                                                                                                       #
# Indian pines --> 16 classes plus background                                                           #
#    --> '../multi_data/MultispectralDatasets/Indian_pines_corrected.mat', 'indian_pines_corrected'     #
#    --> '../multi_data/MultispectralDatasets/Indian_pines_gt.mat', 'indian_pines_gt'                   #
#                                                                                                       #
# KSC --> 13 classes plus background                                                                    #
#    --> '../multi_data/MultispectralDatasets/KSC.mat', 'KSC'                                           #
#    --> '../multi_data/MultispectralDatasets/KSC_gt.mat', 'KSC_gt'                                     #
#                                                                                                       #
# Pavia --> 9 classes plus background                                                                   #
#    --> '../multi_data/MultispectralDatasets/Pavia.mat', 'pavia'                                       #
#    --> '../multi_data/MultispectralDatasets/Pavia_gt.mat', 'pavia_gt'                                 #
#                                                                                                       #
# PaviaU --> 9 classes plus background                                                                  #
#    --> '../multi_data/MultispectralDatasets/PaviaU.mat', 'paviaU'                                     #
#    --> '../multi_data/MultispectralDatasets/PaviaU_gt.mat', 'paviaU_gt'                               #
#                                                                                                       #
# Salinas --> 16 classes plus background                                                                #
#    --> '../multi_data/MultispectralDatasets/Salinas_corrected.mat', 'salinas_corrected'               #
#    --> '../multi_data/MultispectralDatasets/Salinas_gt.mat', 'salinas_gt'                             #
#########################################################################################################  
    
    n_classes = 16
    
    dataset_mat = '../MultispectralDatasets/Indian_pines_corrected.mat'
    dataset_dict = 'indian_pines_corrected' 
    labels_mat = '../MultispectralDatasets/Indian_pines_gt.mat'
    labels_dict = 'indian_pines_gt'
    
    window_sz = 5
    samples = 50
    hidden = 75
    rank = 1
    
    dataset_matrix, label_vector, dataset_matrix_r, label_vector_r, n_components, rpca = LoadData.preprocess_data(
                                                    dataset_mat, dataset_dict,
                                                    labels_mat, labels_dict, 
                                                    window_sz, pca=False,
                                                    noise_percentage=0.2, noise=False)
  
    
    datasets = LoadData.load_data_multi_samples(dataset_matrix_r, label_vector_r, n_components, n_classes, samples)
    train_set_x, train_set_y = datasets[0][0], datasets[0][1]
    test_set_x, test_set_y = datasets[1][0], datasets[1][1]

    
    train_set_x, scaler = PCA.scale_fit(train_set_x)
    test_set_x = PCA.scale_transform(test_set_x, scaler)
    
    batches = LoadData.batch_creation(train_set_x, train_set_y, batch_size=65)
    
    del datasets, dataset_matrix, label_vector, dataset_matrix_r, label_vector_r, dataset_mat, dataset_dict, labels_mat, labels_dict
    
    x_0 = set_tensor_matricization(train_set_x, mode=0)
    x_1 = set_tensor_matricization(train_set_x, mode=1)
    x_2 = set_tensor_matricization(train_set_x, mode=2)
    
    W_0 = torch.empty(hidden, rank, window_sz) # Initialize with xavier
    W_1 = torch.empty(hidden, rank, window_sz)
    W_2 = torch.empty(hidden, rank, n_components)
    V = torch.empty(n_classes, hidden)
    b = torch.zeros(n_classes)
    
    nn.init.xavier_uniform_(W_0)
    nn.init.xavier_uniform_(W_1)
    nn.init.xavier_uniform_(W_2)
    nn.init.xavier_uniform_(V)
    
    
    net_mode_0 = Net_modes(W_0, V, b, n_classes, hidden)
    net_mode_1 = Net_modes(W_1, V, b, n_classes, hidden)
    net_mode_2 = Net_modes(W_2, V, b, n_classes, hidden)
    net_hidden = Net_hidden(V, b, n_classes, hidden)
    
    best_test_acc = 0.0
    for tensor_epochs in range(50):
        
        for batch in batches:
            inputs, labels = batch[0], torch.from_numpy(batch[1])    
            #print("Mode 0")
            x_0 = set_tensor_matricization(inputs, mode=0)
            W_2 = net_mode_2.W
            W_1 = net_mode_1.W
            krp_21 = compute_khatri_rao(W_1.detach().numpy(), W_2.detach().numpy())
            X_0 = torch.from_numpy(np.rollaxis(np.dot(x_0, krp_21), 2, 1))
        
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([net_mode_0.W], lr=0.0005)
                        
            for epochs in range(10):
                
                optimizer.zero_grad()
                outputs = net_mode_0(X_0)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        for batch in batches:
            inputs, labels = batch[0], torch.from_numpy(batch[1])
            #print("Mode 1")
            x_1 = set_tensor_matricization(inputs, mode=1)
            W_2 = net_mode_2.W
            W_0 = net_mode_0.W
            krp_20 = compute_khatri_rao(W_0.detach().numpy(), W_2.detach().numpy())
            X_1 = torch.from_numpy(np.rollaxis(np.dot(x_1, krp_20), 2, 1))
        
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([net_mode_1.W], lr=0.0005)
            
            for epochs in range(10):
                
                optimizer.zero_grad()
                outputs = net_mode_1(X_1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        for batch in batches:
            inputs, labels = batch[0], torch.from_numpy(batch[1])
            #print("Mode 2")
            x_2 = set_tensor_matricization(inputs, mode=2)
            W_1 = net_mode_1.W
            W_0 = net_mode_0.W
            krp_10 = compute_khatri_rao(W_0.detach().numpy(), W_1.detach().numpy())
            X_2 = torch.from_numpy(np.rollaxis(np.dot(x_2, krp_10), 2, 1))
        
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([net_mode_2.W], lr=0.0005)
            
            for epochs in range(10):
                
                optimizer.zero_grad()
                outputs = net_mode_2(X_2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        for batch in batches:
            inputs, labels = batch[0], torch.from_numpy(batch[1])
            #print("Hidden layer")
            W_2 = net_mode_2.W
            W_1 = net_mode_1.W
            W_0 = net_mode_0.W
            krp_321 = compute_khatri_rao_3(W_0.detach().numpy(), W_1.detach().numpy(), W_2.detach().numpy())
            krp_321 = krp_321.sum(axis=2)
            x_vec = set_tensor_vectorization(inputs)
            X_vec = torch.from_numpy(np.dot(x_vec, krp_321.transpose()))
            X_vec = torch.tanh(X_vec)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net_hidden.parameters(), lr=0.0005)
            
            for epochs in range(10):
                
                optimizer.zero_grad()
                outputs = net_hidden(X_vec)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            
            with torch.no_grad():
                temp = copy.deepcopy(net_hidden.fc1.weight)
                net_mode_0.fc2.weight = torch.nn.Parameter(temp)
                net_mode_1.fc2.weight = torch.nn.Parameter(temp)
                net_mode_2.fc2.weight = torch.nn.Parameter(temp)
                
                temp = copy.deepcopy(net_hidden.fc1.bias)                
                net_mode_0.fc2.bias = torch.nn.Parameter(temp)
                net_mode_1.fc2.bias = torch.nn.Parameter(temp)
                net_mode_2.fc2.bias = torch.nn.Parameter(temp)
        
        #print("Epoch : ", tensor_epochs)
        if (tensor_epochs+1)%1 == 0:
            x_vec_test = set_tensor_vectorization(test_set_x)
            X_vec_test = torch.from_numpy(np.dot(x_vec_test, krp_321.transpose()))
            X_vec_test = torch.tanh(X_vec_test)
            
            x_vec = set_tensor_vectorization(train_set_x)
            X_vec = torch.from_numpy(np.dot(x_vec, krp_321.transpose()))
            X_vec = torch.tanh(X_vec)
            
            outputs_train = net_hidden(X_vec)
            output_numpy = torch.argmax(outputs_train, dim=1).numpy()
            print("Epoch : ", tensor_epochs+1)
            print("Training set accuracy : ", accuracy_score(output_numpy, train_set_y))
                
            outputs_test = net_hidden(X_vec_test)
            output_numpy = torch.argmax(outputs_test, dim=1).numpy()
            test_acc = accuracy_score(output_numpy, test_set_y)
            print("....Testing set accuracy : ", test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            print("........Best testing set accuracy : ", best_test_acc)
                