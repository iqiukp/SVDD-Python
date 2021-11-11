# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting with negataive samples

"""
import sys
sys.path.append("..")
from sklearn.datasets import load_wine
from src.BaseSVDD import BaseSVDD, BananaDataset

# Banana-shaped dataset generation and partitioning
X, y = BananaDataset.generate(number=100, display='on')
X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)

# 
svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')

# 
svdd.fit(X_train,  y_train)

# 
svdd.plot_boundary(X_train,  y_train)

#
y_test_predict = svdd.predict(X_test, y_test)

#
radius = svdd.radius
distance = svdd.get_distance(X_test)
svdd.plot_distance(radius, distance)