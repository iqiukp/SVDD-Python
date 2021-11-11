# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting with negataive samples

"""
import sys
sys.path.append("..")
import numpy as np
from src.BaseSVDD import BaseSVDD

# create 100 points with 2 dimensions
n = 100
dim = 2
X = np.r_[np.random.randn(n, dim)]

# svdd object using rbf kernel
svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')

# fit the SVDD model
svdd.fit(X)

# predict the label
y_predict = svdd.predict(X)

# plot the boundary
svdd.plot_boundary(X)

# plot the distance
radius = svdd.radius
distance = svdd.get_distance(X)
svdd.plot_distance(radius, distance)