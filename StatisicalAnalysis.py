# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:51:33 2016
Propose: statistical analysis of one-day spectrum data by hour

@author: kevin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load csv file for channelstate
res=pd.read_csv('DayChannelState.csv')
cs=np.array(res).astype('float')

#calculate the occupy rate
def occbyhour(hcs):
    [hrow,hcol]=hcs.shape
    totsam=hrow*hcol
    cssum=hcs.sum()
    return cssum/totsam

#splite data by hour
[rown,coln]=cs.shape
csbyh=list()
#the first 23 hours as an integer
cspart=cs[0:276:,]
csbyh=np.split(cspart,23,axis=0)
#append the remain 1 hour
csbyh.append(cs[277:(rown+1):,])

#calculate occ rate by hour
occarr=np.arange(24,dtype=np.float)
index=0
for hcs in csbyh:
    tmpocc=occbyhour(hcs)
    occarr[index]=tmpocc
    index=index+1
    #occarr.append(tmpocc)
    print(tmpocc)
    
#visualize the daily vary
plt.plot(occarr)

#test part
tarr=np.array([[3,4],[5,6]])
tsum=tarr.sum()



