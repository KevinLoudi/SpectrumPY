# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:51:33 2016
Propose: statistical analysis of one-day spectrum data by hour

@author: kevin
"""

import scipy
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitter import Fitter


#load csv file for channelstate
res=pd.read_csv('DayChannelState.csv')
cs=np.array(res).astype('float')

#calculate the occupy rate
def occbyhour(hcs):
    [hrow,hcol]=hcs.shape
    totsam=hrow*hcol
    cssum=hcs.sum()
    return cssum/totsam
    
#find and count spectrum-hole
def holesearcher(cs):
    [tlen,flen]=cs.shape
    #the possible maxiest hole-length
    mhole=tlen
    #decalre an array to store hole num
    hole=np.arange(mhole+1,dtype=np.float)*0
    lenindex=0
    for f in np.arange(flen):
      #if all time state in a freq-col equals '0'
        if(lenindex!=0):
           hole[lenindex]=hole[lenindex]+1
           lenindex=0  
        for t in np.arange(tlen):
            if(cs[t,f]==0):
                lenindex=lenindex+1 
            else:
                hole[lenindex]=hole[lenindex]+1
                lenindex=0 
    return hole
    
#calculate possible unoccpuied state time-length distribution
def holelendis(hole):
    tlen=len(hole)
    totlen=hole.sum()
    hish=np.arange(tlen,dtype=np.float)*0
    for t in np.arange(tlen):
        hish[t]=hole[t]/totlen
    return hish
        

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
#plt.plot(occarr)

#fit data emprical distribution by-hour colwise
hh=holesearcher(cs)
hish=holelendis(hh)
plt.plot(hish)
plt.xlim(0,47)

#test part
tarr=np.array([[3,4],[5,6]])
tsum=tarr.sum()
tan=np.arange(10)



