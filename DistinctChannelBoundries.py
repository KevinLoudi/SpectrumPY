# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:50:41 2016
Propose：Distinct Channel Boundry through K-means clustering
@author: kevin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

#load csv file for channelstate

#use a fixed threshold
#res=pd.read_csv('DayChannelState.csv')
#cs=np.array(res).astype('float')

#use clustering method to adoptivly decided the threshold
res=pd.read_csv('DayLevel.csv')
lev=np.array(res).astype('float')

#cluster act
def doclusterto2class(data):
    # computing K-Means with K = 2 (2 clusters)
    centroids,_ = kmeans(data,2)
    # assign each sample to a cluster
    idx,_ = vq(data,centroids)
    return idx #np.transpose(idx)

#clustering by time slice
def clusterbytime(lev):
    [row,col]=lev.shape
    tlevave=np.transpose(np.arange(col)*0.0)
    tlevmax=tlevave
    tlevmin=tlevave
    for i in np.arange(row):
        #cluster act,return a classify label "channel" or "dumy"
        #do clustering for each time slot
        tmprow=doclusterto2class(lev[i,:])
        tlevave=tlevave+tmprow
        for j in np.arange(col):
            if tlevmax[j]<tmprow[j]:   
               tlevmax[j]=tmprow[j]
            if tlevmin[j]>tmprow[j]:
               tlevmin[j]=tmprow[j] 
    #Here I want to see how like the sample could represent a channel
    tlevave=tlevave/row #the possibilty of being a channel
    return tlevmin,tlevave,tlevmax
    
#search and label possible channel, like "channel1","channel2"...
#input ch label '1' 
def markthechannels(ch,bw):
    
    return 0
    
#get the channel mask
[lmin,lmean,lmax]=clusterbytime(lev)
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
[row,col]=lev.shape
freq=np.arange(col)
plt.plot(lmean) #probability of being a channel
plt.plot(lmin) #valiad channel
#plt.plot(lmax)
plt.show()

    
