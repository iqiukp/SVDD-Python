# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
import scipy.io as sio


# load data
class PrepareData():
    

    def banana():
    
        data = sio.loadmat(".\\data\\banana.mat")
        
        trainData = data['trainData'] # training data 
        testData = data['testData']# testng data
    
        trainLabel = data['trainLabel'] # training data 
        testLabel = data['testLabel'] # testng data
    
        return trainData, testData,trainLabel, testLabel
    
    def iris():
    
        iris = datasets.load_iris()
        data =  preprocessing.StandardScaler().fit_transform(iris.data)
        label = iris.target


        p_data = data[label == 0, :]
        n_data = data[label != 0, :]
        
        p_label = np.mat(np.ones(p_data.shape[0])).T
        n_label = np.mat(-np.ones(n_data.shape[0])).T
        
        
        p_x, p_xt, p_y, p_yt = train_test_split(p_data, p_label, test_size=0.3, random_state=1)
        n_x, n_xt, n_y, n_yt = train_test_split(n_data, n_label, test_size=0.9, random_state=1)
        
        trainData = np.vstack((p_x, n_x))
        testData = np.vstack((p_xt, n_xt))
        trainLabel = np.vstack((p_y, n_y))
        testLabel = np.vstack((p_yt, n_yt))
    
        return trainData, testData,trainLabel, testLabel

    def TE():
    
        data = sio.loadmat(".\\data\\TE.mat")
        
        trainData = data['trainData'] # training data 
        testData = data['testData']# testng data
    
        trainLabel = data['trainLabel'] # training data 
        testLabel = data['testLabel'] # testng data
    
        return trainData, testData,trainLabel, testLabel

