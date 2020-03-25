# -*- coding: utf-8 -*-

import sys
sys.path.append("..")
from src.svdd import SVDD
from src.visualize import Visualization as draw
from data import PrepareData as load

# load banana-shape data
trainData, testData, trainLabel, testLabel = load.iris()

# set SVDD parameters
parameters = {"positive penalty": 0.9,
              "negative penalty": 0.8,
              "kernel": {"type": 'gauss', "width": 1/24},
              "option": {"display": 'on'}}


# construct an SVDD model
svdd = SVDD(parameters)

# train SVDD model
svdd.train(trainData, trainLabel)


# test SVDD model
distance, accuracy = svdd.test(testData, testLabel)

# visualize the results
draw.testResult(svdd, distance)
draw.testROC(testLabel, distance)

