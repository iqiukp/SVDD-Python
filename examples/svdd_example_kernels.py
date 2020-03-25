# -*- coding: utf-8 -*-


import sys
sys.path.append("..")
from src.svdd import SVDD
from src.visualize import Visualization as draw
from data import PrepareData as load

# load banana-shape data
trainData, testData, trainLabel, testLabel = load.banana()


# kernel list
kernelList = {"1": {"type": 'gauss', "width": 1/24},
              "2": {"type": 'linear', "offset": 0},
              "3": {"type": 'ploy', "degree": 2, "offset": 0},
              "4": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
              "5": {"type": 'lapl', "width": 1/12}
              }


for i in range(len(kernelList)):

    # set SVDD parameters
    parameters = {"positive penalty": 0.9,
                  "negative penalty": 0.8,
                  "kernel": kernelList.get(str(i+1)),
                  "option": {"display": 'on'}}
    
    # construct an SVDD model
    svdd = SVDD(parameters)
    
    # train SVDD model
    svdd.train(trainData, trainLabel)
      
    # test SVDD model
    distance, accuracy = svdd.test(testData, testLabel)
    
    # visualize the results
    # draw.testResult(svdd, distance)
    # draw.testROC(testLabel, distance)
    draw.boundary(svdd, trainData, trainLabel)



