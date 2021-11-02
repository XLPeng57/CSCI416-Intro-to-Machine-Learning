import numpy as np
import math

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    p = 1 / (1 + np.exp(np.negative(z)))
    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE

    theta_x = X.dot(theta.T)
    predict = logistic(theta_x)
    cost = np.dot(-y,np.log(predict)) - np.dot((1-y), np.log(1-predict))
    return cost

def gradient_descent( X, y, theta, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    # REPLACE CODE BELOW WITH CORRECT CODE
    m,n = X.shape
    
    if theta is None:
        theta = np.zeros(n)
    
    # For recording cost function value during gradient descent
    J_history = np.zeros(iters)

    for i in range(0, iters):
        
        # TODO: compute gradient (vectorized) and update theta
        theta_x = X.dot(theta.T)
        prediction = logistic(theta_x)
        theta = theta - alpha*(X.T.dot((prediction - y)))

        # Record cost function
        J_history[i] = cost_function(X, y, theta)
        
    return theta, J_history
