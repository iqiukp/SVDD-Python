# -*- coding: utf-8 -*-
"""

An example for parameter selection using grid search

"""
import sys
sys.path.append("..")
from sklearn.datasets import load_wine
from src.BaseSVDD import BaseSVDD, BananaDataset
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.model_selection import learning_curve, GridSearchCV

# Banana-shaped dataset generation and partitioning
X, y = BananaDataset.generate(number=100, display='off')
X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)

param_grid = [
    {"kernel": ["rbf"], "gamma": [0.1, 0.2, 0.5], "C": [0.1, 0.5, 1]},
    {"kernel": ["linear"], "C": [0.1, 0.5, 1]},
    {"kernel": ["poly"], "C": [0.1, 0.5, 1], "degree": [2, 3, 4, 5]},
]

svdd = GridSearchCV(BaseSVDD(display='off'), param_grid, cv=5, scoring="accuracy")
svdd.fit(X_train, y_train)
print("best parameters:")
print(svdd.best_params_)
print("\n")

# 
best_model = svdd.best_estimator_
means = svdd.cv_results_["mean_test_score"]
stds = svdd.cv_results_["std_test_score"]

for mean, std, params in zip(means, stds, svdd.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()
