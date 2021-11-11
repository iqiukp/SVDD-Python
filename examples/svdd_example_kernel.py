# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting using different kernels

"""
import sys
sys.path.append("..")
from src.BaseSVDD import BaseSVDD, BananaDataset

# Banana-shaped dataset generation and partitioning
X, y = BananaDataset.generate(number=100, display='on')
X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)

# kernel list
kernelList = {"1": BaseSVDD(C=0.9, kernel='rbf', gamma=0.3, display='on'),
              "2": BaseSVDD(C=0.9, kernel='poly',degree=2, display='on'),
              "3": BaseSVDD(C=0.9, kernel='linear', display='on')
              }

# 
for i in range(len(kernelList)):
    svdd = kernelList.get(str(i+1))
    svdd.fit(X_train,  y_train)
    svdd.plot_boundary(X_train,  y_train)




