# -*- coding: utf-8 -*-
"""
An example for cross validation

"""
import sys
sys.path.append("..")
from src.BaseSVDD import BaseSVDD, BananaDataset
from sklearn.model_selection import cross_val_score


# Banana-shaped dataset generation and partitioning
X, y = BananaDataset.generate(number=100, display='on')
X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)

# 
svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')


# cross validation (k-fold)
k = 5
scores = cross_val_score(svdd, X_train, y_train, cv=k, scoring='accuracy')

#
print("Cross validation scores:")
for scores_ in scores:
    print(scores_)
 
print("Mean cross validation score: {:4f}".format(scores.mean()))

