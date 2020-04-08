# -*- coding: utf-8 -*-


import numpy as np
from cvxopt import matrix, solvers
import sklearn.metrics.pairwise as smp
import time

class SVDD():
    
    def __init__(self, parameters):
        
        """ 
        DESCRIPTION
        
        --------------------------------------------------        
        INPUT
          parameters   

             "positive penalty": positive penalty factor
             "negative penalty": negative penalty factor
             "kernel"          : kernel function         
             "option"          : some options 
             
        
        """                
        self.parameters = parameters



    def train(self, data, label):
        
        """ 
        DESCRIPTION
        
        Train SVDD model
        
        -------------------------------------------------- 
        Reference
        Tax, David MJ, and Robert PW Duin.
        "Support vector data description." 
        Machine learning 54.1 (2004): 45-66.
        
        -------------------------------------------------- 
        model = train(data, label)
        
        --------------------------------------------------        
        INPUT
        data        Training data (n*d) 
                        n: number of samples
                        d: number of features
        label       Training label (n*1)
                        positive: 1
                        negative: -1
                        
        OUTPUT
        model       SVDD hypersphere
        --------------------------------------------------
        
        """
        start_time = time.time()
        
        label = np.array(label, dtype = 'int')      
        if np.abs(np.sum(label)) == data.shape[0]:
            self.labeltype = 'single'
        else:
            self.labeltype = 'hybrid'
        
        # index of positive and negative samples
        pIndex = label[:,0] == 1
        nIndex = label[:,0] == -1
        
        # threshold for support vectors
        threshold = 1e-7
        
        # compute the kernel matrix
        K = self.getMatrix(data, data)

        # solve the Lagrange dual problem
        alf, obj, iteration = self.quadprog(K, label)
        
        # find the index of support vectors
        sv_index = np.where(alf > threshold)[0][:]

        # support vectors and alf
        sv_value = data[sv_index, :]
        sv_alf = alf[sv_index]
        
        # compute the center of initial feature space
        center = np.dot(alf.T, data)
        
        ''''
        compute the radius: eq(15)
        
        The distance from any support vector to the center of 
        the sphere is the hypersphere radius. 
        Here take the 1st support vector as an example.
        
        '''
        # the 1st term in eq(15)
        used = 0
        term_1 = K[sv_index[used], sv_index[used]]
        
        # the 2nd term in eq(15)
        term_2 = -2*np.dot(K[sv_index[used], :], alf)
        
        # the 3rd term in eq(15)
        term_3 = np.dot(np.dot(alf.T, K), alf)

        # R
        radius = np.sqrt(term_1+term_2+term_3)
        
        end_time = time.time()
        timecost = end_time - start_time
        
        # numbers of positive and negative samples
        pData = np.sum(pIndex)/data.shape[0]
        nData = np.sum(nIndex)/data.shape[0]
        
        # number of support vectors
        nSVs = sv_index.shape[0]
        
        # radio of  support vectors
        rSVs = nSVs/data.shape[0]
        
        # store the results
        self.model = {"data"      : data        ,
                      "sv_alf"    : sv_alf      ,
                      "radius"    : radius      ,
                      "sv_value"  : sv_value    ,
                      "sv_index"  : sv_index    ,
                      "nSVs"      : nSVs        ,
                      "center"    : center      ,
                      "term_3"    : term_3      ,
                      "alf"       : alf         ,  
                      "K"         : K           ,
                      "pIndex"    : pIndex      ,
                      "nIndex"    : nIndex      ,
                      "obj"       : obj         ,
                      "iteration" : iteration   ,
                      "timecost"  : timecost    ,
                      "pData"     : pData       ,
                      "nData"     : nData       ,
                      "rSVs"      : rSVs        ,
                      }
        
        # compute the training accuracy
        display_ = self.parameters["option"]["display"]
        self.parameters["option"]["display"] = 'off'
        _, accuracy = self.test(data, label)
        self.parameters["option"]["display"] = display_      
        self.model["accuracy"] = accuracy
        
        # display training results       
        if self.parameters["option"]["display"] == 'on':
            print('\n')
            print('*** SVDD model training finished ***\n')
            print('iter             = %d'       % self.model["iteration"])
            print('time cost        = %.4f s'   % self.model["timecost"])
            print('obj              = %.4f'     % self.model["obj"])
            print('pData            = %.4f %%'  % (100*self.model["pData"]))
            print('nData            = %.4f %%'  % (100*self.model["nData"]))
            print('nSVs             = %d'       % self.model["nSVs"])
            print('radio of nSVs    = %.4f %%'  % (100*self.model["rSVs"]))
            print('accuracy         = %.4f %%'  % (100*self.model["accuracy"]))
            print('\n')
  
    def test(self, data, label):
    
        """ 
        DESCRIPTION
        
        Test the testing data using the SVDD model
    
        distance = test(model, Y)
        
        --------------------------------------------------        
        INPUT
        data        Test data (n*d) 
                        n: number of samples
                        d: number of features
        label       Test label (n*1)
                        positive: 1
                        negative: -1
            
        OUTPUT
        distance    Distance between the test data and hypersphere
        --------------------------------------------------
        
        """    
        
        start_time = time.time()
        n = data.shape[0]
        
        # compute the kernel matrix
        K = self.getMatrix(data, self.model["data"])
        
        # the 1st term
        term_1 = self.getMatrix(data, data)
        
        # the 2nd term
        tmp_1 = -2*np.dot(K, self.model["alf"])
        term_2 = np.tile(tmp_1, (1, n))
        
        # the 3rd term
        term_3 =  self.model["term_3"]
        
        # distance
        distance = np.sqrt(np.diagonal(term_1+term_2+term_3))
        
        # predicted label
        predictedlabel = np.mat(np.ones(n)).T
        fault_index = np.where(distance > self.model["radius"])[1][:]
        predictedlabel[fault_index] = -1
            
        # compute prediction accuracy
        accuracy = np.sum(predictedlabel == label)/n
        
        end_time = time.time()
        timecost = end_time - start_time
        if self.parameters["option"]["display"] == 'on':
        # display test results
            print('\n')
            print('*** SVDD model test finished ***\n')
            print('time cost        = %.4f s'   % timecost)
            print('accuracy         = %.4f %%'  % (100*accuracy))
            print('\n')
        
        
        return distance, accuracy 

    def quadprog(self, K, label):
    
        """ 
        DESCRIPTION
        
        Solve the Lagrange dual problem
        
        quadprog(self, K, label)
        
        --------------------------------------------------
        INPUT
        K         Kernel matrix
        label     training label
        
                        
        OUTPUT
        alf       Lagrange multipliers
        
        --------------------------------------------------
        
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b                    
        --------------------------------------------------
        
        """ 
        solvers.options['show_progress'] = False
        
        label = np.mat(label)
        K = np.multiply(label*label.T, K)
        
        # P
        n = K.shape[0]
        P = K+K.T
        
        # q
        q = -np.multiply(label, np.mat(np.diagonal(K)).T)

        # G
        G1 = -np.eye(n)
        G2 = np.eye(n)
        G = np.append(G1, G2, axis=0)
        
        # h
        h1 = np.mat(np.zeros(n)).T # lb
        h2 = np.mat(np.ones(n)).T
        if self.labeltype == 'single':
            h2[label == 1] = self.parameters["positive penalty"]
        
        if self.labeltype == 'hybrid':
            h2[label == 1] = self.parameters["positive penalty"]
            h2[label == -1] = self.parameters["negative penalty"]

            
        h = np.append(h1, h2, axis=0)
        
        # A, b
        A = np.mat(np.ones(n))
        b = 1.
        
        #
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        #
        sol =solvers.qp(P, q, G, h, A, b)
        alf = np.array(sol['x'])
        obj = np.array(sol['dual objective'])
        iteration = np.array(sol['iterations'])

        return alf, obj, iteration

    def getMatrix(self, X, Y):
    
        """ 
        DESCRIPTION
        
        Compute kernel matrix 
        
        K = getMatrix(X, Y)
        
        -------------------------------------------------- 
        INPUT
        X         data (n*d)
        Y         data (m*d)

        OUTPUT
        K         kernel matrix 
        -------------------------------------------------- 
                        
                            
        type   -  
        
        linear :  k(x,y) = x'*y+c
        poly   :  k(x,y) = (x'*y+c)^d
        gauss  :  k(x,y) = exp(-s*||x-y||^2)
        tanh   :  k(x,y) = tanh(g*x'*y+c)
        lapl   :  k(x,y) = exp(-s*||x-y||)
           
        degree -  d
        offset -  c
        width  -  s
        gamma  -  g
        
        --------------------------------------------------      
        ker    - 
        
        ker = {"type": 'gauss', "width": s}
        ker = {"type": 'linear', "offset": c}
        ker = {"type": 'ploy', "degree": d, "offset": c}
        ker = {"type": 'tanh', "gamma": g, "offset": c}
        ker = {"type": 'lapl', "width": s}
    
        """
        def gaussFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.rbf_kernel(X, Y, gamma=s)

                
            return K
            
        def linearFunc():
            
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0

            K = smp.linear_kernel(X, Y)+c
            
            return K
        
        def ployFunc():
            if self.parameters["kernel"].__contains__("degree"):
                d =  self.parameters["kernel"]["degree"]
            else:
                d = 2
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0
                
            K = smp.polynomial_kernel(X, Y, degree=d, gamma=None, coef0=c)
            
            return K
             
        def laplFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.laplacian_kernel(X, Y, gamma=s)

            return K
    
        def tanhFunc():
            if self.parameters["kernel"].__contains__("gamma"):
                g =  self.parameters["kernel"]["gamma"]
            else:
                g = 0.01
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 1
            
            K = smp.sigmoid_kernel(X, Y, gamma=g, coef0=c)

            return K

        kernelType = self.parameters["kernel"]["type"]
        switcher = {    
                        "gauss"   : gaussFunc  ,        
                        "linear"  : linearFunc ,
                        "ploy"    : ployFunc   ,
                        "lapl"    : laplFunc   ,
                        "tanh"    : tanhFunc   ,
                     }
        
        return switcher[kernelType]()
