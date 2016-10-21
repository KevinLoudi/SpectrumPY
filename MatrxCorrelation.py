# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:23:47 2016
Propose:Computing the correlation coefficient between two multi-dimensional arrays
@author: kevin
"""
"""reserve for direct console input
 A = np.random.rand(1000,100)
 B = np.random.rand(1000,100)
 A_mA = A - A.mean(1)[:,None]
 B_mB = B - B.mean(1)[:,None]
"""

import numpy as np
from scipy.stats import pearsonr

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    
def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
    
def test_generate_correlation_map():
    x = np.random.rand(10, 10)
    y = np.random.rand(20, 10)
    desired = np.empty((10, 20))
    for n in range(x.shape[0]):
        for m in range(y.shape[0]):
            desired[n, m] = pearsonr(x[n, :], y[m, :])[0]
    actual = generate_correlation_map(x, y)
    #Raises an AssertionError if two objects are not equal up to desired precision.
    np.testing.assert_array_almost_equal(actual, desired)
    
    
test_generate_correlation_map()

#A = np.random.rand(1000,100)
#B = np.random.rand(1000,100)
#
#X=corr2_coeff(A,B)
#
#Y=generate_correlation_map(A, B)