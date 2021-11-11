# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting using nonlinear principal component.

The KPCA algorithm is used to reduce the dimension of the original data.

"""

import sys
sys.path.append("..")
import numpy as np
from src.BaseSVDD import BaseSVDD
from sklearn.decomposition import KernelPCA


# create 100 points with 5 dimensions
X = np.r_[np.random.randn(50, 5) + 1, np.random.randn(50, 5)]
y = np.append(np.ones((50, 1), dtype=np.int64), 
              -np.ones((50, 1), dtype=np.int64),
              axis=0)

# number of the dimensionality
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)

# fit the SVDD model
svdd = BaseSVDD(C=0.9, gamma=10, kernel='rbf', display='on')

# fit and predict
svdd.fit(X_kpca,  y)
y_test_predict = svdd.predict(X_kpca, y)

# plot the distance curve
radius = svdd.radius
distance = svdd.get_distance(X_kpca)
svdd.plot_distance(radius, distance)

# plot the boundary
svdd.plot_boundary(X_kpca,  y)

