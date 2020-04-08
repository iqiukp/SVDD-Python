# Support Vector Data Description (SVDD)

Python Code for abnormal detection or fault detection using SVDD.
    
Email: iqiukp@outlook.com

-------------------------------------------------------------------

## Main features

* SVDD model for training dataset containing only positive training data. (SVDD)
* SVDD model for training dataset containing both positive training data and negative training data. (nSVDD)
* Multiple kinds of kernel functions.
* Visualization module including ROC curve plotting, test result plotting, and decision boundary.

-------------------------------------------------------------------

## Requirements

* matplotlib
* cvxopt
* scipy
* numpy
* scikit_learn

-------------------------------------------------------------------

## About SVDD model

Two types of SVDD models are built according to the following references:

[1]    Tax D M J, Duin R P W. Support vector data description[J]. Machine learning, 2004, 54(1): 45-66.

-------------------------------------------------------------------

## A simple application for decision boundary (using differnent kernel functions)

```

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

```

* gaussian kernel function

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/kernel_gauss.png" width="720">
</p>

* linear kernel function

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/kernel_linear.png" width="720">
</p>

* polynomial kernel function

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/kernel_ploy.png" width="720">
</p>

* sigmoid kernel function

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/kernel_tanh.png" width="720">
</p>

* laplacian kernel function

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/kernel_lapl.png" width="720">
</p>


## A simple application for abnormal detection or fault detection

```

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

```

* test result

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/hybrid_result.png" width="720">
</p>

* ROC curve

<p align="middle">
  <img src="https://github.com/iqiukp/SVDD/blob/master/imgs/hybrid_roc.png" width="480">
</p>

