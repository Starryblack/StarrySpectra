#!/usr/bin/env python3
import math
import numpy as np
from scipy import sparse
from numpy import array
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import sys
import argparse

## Works off Iterative Weighted Least Squares (IWLS) (Automated Background Subtraction Algorithm for Raman Spectra Based on Iterative Weighted Least Squares)

class SuiteRaman():

    def __init__(self):
        print('Hello World!')

    def parsefile(self, filename):
        self.filename = filename
        _data = np.loadtxt('{}'.format(self.filename))
        self.x = _data[:, 0]
        self.x = x[0:]
        self.y = _data[:, 1]
        self.y = y[0:]
        self.x = np.array([x]).T
        self.y = np.array([y]).T
        self.y = np.asmatrix(y)
        self.length = len(x)
        print('Parsing...done.')

    def ramanbaseline(self, polyorder = 3, residual = 0.001, coarseness = 0.0001, window = 50, iterations = 50):

        vander = np.ones((self.length,1))
        for i in range(1, polyorder+1):
            xpower = np.power(x, i)
            vander = np.append(vander, xpower, axis=1)
            print(vander, np.shape(vander))
        vander = np.asmatrix(vander)
        print("Vandermonde matrix generated...")

        #Calculate coarseness c (L x 1) via a moving window standard deviation of the first derivative
        #rolling std and mean function
        def rocknroll(input, window):
            temp = np.empty(input.shape)
            temp.fill(np.nan)
            for i in range(window - 1, input.shape[0]):
                q = input[(i-window+1):i+1]
                temp[i-window+1] = np.sqrt((1/(window-1))*np.mean(np.power((q - q.mean()),2)))

            for i in range(input.shape[0] - window + 1, input.shape[0]):
                q = input[i:]
                temp[i] = np.sqrt((1/(window-1))*np.mean(np.power((q - q.mean()),2)))

            return temp

        #actual iteration process for fitness parameter p, threshold parameter n, number of iterations niter

        for t in range(1, iterations+1):
            print("Iteration ", t)
            if t == 1:
                #initialise weight matrix using nxn identity matrix co
                weight = np.identity(self.length, dtype=np.float64)
                prevco = np.zeros((polyorder+1,1))
            else:
                prevco = co
                weight = np.multiply((r < 0) + residual*(r > 0), ((coarsenormalised < coarseness) + coarseness*(coarsenormalised > coarseness)))
                weight = np.diagflat(weight)
                #print (weight)
            #initialise coefficient vector ((n+1) x 1)
            co = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul((np.transpose(vander)),weight), vander)), np.transpose(vander)), weight), y)
            #print(co)
            rms = math.sqrt(np.mean(np.power((co - prevco), 2)))
            print('RMS = ', rms, "\n")
            r = y - np.matmul(vander, co)
            d = np.gradient(r, axis=0)
            finalcoarse = rocknroll(d, window)
            coarsenormalised = (finalcoarse - min(coarse))/(max(coarse)-min(coarse))

            if rms == 0:
                break

        self.residual = np.matmul(vander, co)
        self.signal = y - self.residual

        baseddata = np.append(x, signal, axis=1)
        np.savetxt('BASED_{}.txt'.format(self.filename), baseddata, fmt=['%.2f','%.3f'], delimiter='\t')

    def plotraw(self, xlabel, ylabel):

        fig = plt.figure()
        fig.suptitle('RAW_{}'.format(self.filename), fontsize=15)
        plt.xlabel('{}'.format(xlabel), fontsize=15)
        plt.ylabel('{}'.format(ylabel), fontsize=15)
        plt.plot(self.x, self.y)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def plotbased(self, xlabel, ylabel):

        self.ramanbaseline()
        fig = plt.figure()
        fig.suptitle('BASED_{}'.format(self.filename), fontsize=15)
        plt.xlabel('{}'.format(xlabel), fontsize=15)
        plt.ylabel('{}'.format(ylabel), fontsize=15)
        plt.plot(self.x, self.y)
        plt.plot(self.x, self.signal)
        plt.plot(self.x, self.residual)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        pass

##
