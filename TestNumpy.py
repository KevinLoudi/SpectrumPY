# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 13:48:05 2016

@author: kevin

TODO: 
"""

import scipy.misc
import matplotlib.pyplot as plt

lena=scipy.misc.lena()
xmax=lena[0]
ymax=lena[1]
lena[range(xmax),range(ymax)]=0
