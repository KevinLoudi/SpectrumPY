# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:54:39 2016

@author: kevin
"""

import theano.tensor as T
import numpy as np
from theano import function
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y

#make a simple function to add two number together
f=function([x,y],z)
print(f(2,3))
np.allclose(f(16.3,12.1),28.4)