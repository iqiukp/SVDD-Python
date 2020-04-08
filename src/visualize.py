# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from mpl_toolkits.mplot3d import Axes3D
import time

class Visualization():
    
    def testResult(svdd, distance):
    
        """ 
        DESCRIPTION
        
        Plot the test results
        
        testResult(model, distance)
        
        --------------------------------------------------------------- 
        
        INPUT
        svdd             SVDD hypersphere
        distance         distance from test data to SVDD hypersphere 
        
        --------------------------------------------------------------- 
        
        """
        plt.rcParams['font.size'] = 15
        n = distance.shape[0]
        
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_subplot(1, 1, 1)
        radius = np.ones((n, 1))*svdd.model["radius"]
        ax.plot(radius, 
                color ='r',
                linestyle = '-', 
                marker = 'None',
                linewidth = 2, 
                markeredgecolor ='k',
                markerfacecolor = 'w', 
                markersize = 6)
        
        ax.plot(distance,
                color = 'k',
                linestyle = ':',
                marker='o',
                linewidth=1,
                markeredgecolor = 'k',
                markerfacecolor = 'C4',
                markersize = 6)
        
        ax.set_xlabel('Samples')
        ax.set_ylabel('Distance')
        
        ax.legend(["Radius","Distance"], 
                  ncol = 1, loc = 0, 
                  edgecolor = 'black', 
                  markerscale = 2, fancybox = True)
    
        plt.show() 
        
    def testROC(label, distance):
        """ 
        DESCRIPTION
        
        Plot the test ROC
        
        testROC(label, distance)
        
        --------------------------------------------------------------- 
        
        INPUT
        label            test label
        distance         distance from test data to SVDD hypersphere 
        
        --------------------------------------------------------------- 
        
        """
        if np.abs(np.sum(label)) == label.shape[0]:
            raise SyntaxError('Both positive and negative labels must be entered for plotting ROC curve.')
       
        # number of positive samples
        plt.rcParams['font.size'] = 15
        n_p = np.sum(label == 1)
        n_n = np.sum(label == -1)
        
        #  sort
        index = np.argsort(distance)
        label_sort = label[index]
        FP = np.cumsum(label_sort == -1)
        TP = np.cumsum(label_sort == 1)
        FPR = FP/n_n
        TPR = TP/n_p
        
        roc_auc = auc(FPR.T, TPR.T) 
                  
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(FPR.T, TPR.T,
                color ='C3',
                linestyle = '-', 
                marker = 'None',
                linewidth = 5, 
                markeredgecolor ='k',
                markerfacecolor = 'w', 
                markersize = 6)
        
        ax.set_xlabel('False positive rate (FPR)')
        ax.set_ylabel('True positive rate (TPR)')
        ax.set_title('Area under the curve (AUC) = %.4f' %roc_auc)
        
        plt.grid()
        plt.show()
        
        
    def boundary(svdd, data, label, r=0.3, nn=2):
        """ 
        DESCRIPTION
        
        Plot the boundary
        
        boundary(svdd, data, label, r=0.3, nn=2)
        
        --------------------------------------------------------------- 
        
        INPUT
        svdd             SVDD hypersphere
        data             training data 
        label            training label
        r                radio of expansion (0<r<1)
        nn               number of grids
        
        --------------------------------------------------------------- 
        
        """ 
        
        dim = data.shape[1]
        if dim!=2:
            raise SyntaxError('Visualization of decision boundary only supports for 2D data')
    
        # compute the range of grid 
        numGrids = np.rint(data.shape[0]/nn).astype(int)  # number of grids
        x_range = np.zeros(shape=(numGrids, 2))
        for i in range(2):  
            _tmp_ = (np.max(data[:, i])-np.min(data[:, i]))*r
            xlim_1 = np.min(data[:, i])-_tmp_
            xlim_2 = np.max(data[:, i])+_tmp_
            x_range[:, i] = np.linspace(xlim_1, xlim_2, numGrids)
        
        # grid
        xv, yv = np.meshgrid(x_range[:, 0], x_range[:, 1])
        
        num1 = xv.shape[0]
        num2 = yv.shape[0]
        distance = np.zeros(shape=(num1, num1))
        
        # calculate the grid scores
        print("Calculating the grid (%04d*%04d) scores...\n" %(num1, num2))
        
        display_ = svdd.parameters["option"]["display"]
        svdd.parameters["option"]["display"] = 'off'
        start_time = time.time()       
        for i in range(num1):
            for j in range(num2):
                tmp = np.mat([xv[i, j], yv[i, j]])   
                distance[i, j], _ = svdd.test(tmp, 1)
                # print('[feature 1: %06d]  [feature 2: %06d] \n' % (i+1,j+1))
        end_time = time.time()
        print('Grid scores completed. Time cost %.4f s\n' % (end_time-start_time))
        svdd.parameters["option"]["display"] = display_
        
        # plot the contour (3D)
        fig = plt.figure(figsize = (20, 6))
        
        ax3 = fig.add_subplot(1, 3, 1, projection='3d') 
        # ax3 = ax3.axes(projection='3d')
        ada = ax3.plot_surface(xv, yv, distance, cmap=plt.cm.jet)
        ax3.contourf(xv, yv, distance, zdir='z', offset=np.min(distance)*0.9, cmap=plt.cm.coolwarm)
        ax3.set_zlim(np.min(distance)*0.9, np.max(distance)*1.05)
        # plt.colorbar(ada)
            
    
    
        # plot the contour (2D)
        # fig = plt.figure(figsize = (10, 8))
        ax1 = fig.add_subplot(1, 3, 2)    
          
        ctf1 = ax1.contourf(xv, yv, distance, alpha = 0.8, cmap=plt.cm.jet)
        ctf2 = ax1.contour(xv, yv, distance, colors='black', linewidths=1)
        plt.clabel(ctf2, inline=True)
        # plt.colorbar(ctf1)
        
        # plot the boundary
        # fig = plt.figure(figsize = (10, 8))
        ax2 = fig.add_subplot(1, 3, 3)    
        
        if svdd.labeltype == 'single':

            ax2.scatter(data[:,0], data[:,1],
                        color='green',marker='o',
                        edgecolor='black',alpha=0.5, zorder = 2)
            ax2.scatter(data[svdd.model["sv_index"],0], data[svdd.model["sv_index"],1],
                    facecolor='C2',marker='o',s = 144,linewidths = 2,
                    edgecolor='black', zorder = 2)
        
            ax2.contour(xv, yv, distance,[svdd.model["radius"]],
                              colors='C3', linewidths=5, zorder = 1)
            
            ax2.legend(["Training data", "Support vectors"], 
                      ncol = 1, loc = 0, 
                      edgecolor = 'black',markerscale = 1.2, fancybox = True) 
                
        else:
            ax2.scatter(data[svdd.model["pIndex"],0], data[svdd.model["pIndex"],1],
                    facecolor='C0',marker='o', s = 100,linewidths = 2,
                    edgecolor='black', zorder = 2)
            ax2.scatter(data[svdd.model["nIndex"],0], data[svdd.model["nIndex"],1],
                    facecolor='C4',marker='s',s = 100,linewidths = 2,
                    edgecolor='black', zorder = 2)
        
            ax2.scatter(data[svdd.model["sv_index"],0], data[svdd.model["sv_index"],1],
                    facecolor='C2',marker='o',s = 144,linewidths = 2,
                    edgecolor='black', zorder = 2)
            
            ax2.contour(xv, yv, distance,[svdd.model["radius"]],
                              colors='C3', linewidths=5, zorder = 1)
            
            ax2.legend(["Training data (+)","Training data (-)", "Support vectors"], 
                      ncol = 1, loc = 0, 
                      edgecolor = 'black',markerscale = 1.2, fancybox = True) 
        
        plt.show()
        