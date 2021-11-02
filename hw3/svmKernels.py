"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """

    return (np.dot(X1, X2.T)+1) ** _polyDegree



def myGaussianKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    n1,d1 = X1.shape
    n2,d2 = X2.shape

    dis = np.zeros((n1, n2))
    for i in range(0, n2):
      dis[:,i] = np.sum((X1 - X2[i,:]) ** 2, axis = 1)
    
    temp = 2 * (_gaussSigma ** 2)
    return np.exp(-dis / temp)
