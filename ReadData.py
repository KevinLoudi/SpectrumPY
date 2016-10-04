# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:29:38 2016

@author: kevin

TODO: load spectrum data from .*argus files 
"""



import scipy.io as scio
import numpy as np

#normalize array data to its [min,max]
def MaxMinNormalization(A):
    [row,col]=np.shape(A)
    tmp=np.reshape(A,row*col)
    Max=np.max(tmp)
    Min=np.min(tmp)
    Diff=Max-Min
    X=np.zeros([row,col],float)
    X=1000*(A-Min)/Diff
    return X

#abstract part of the array
def statANA(array,low,high):
  [row,col]=np.shape(array)
  if(low>high) or (high>col):
    return
  newarray=array[:,low:high]
  return (newarray)
  
#change continous array data to states 0, 1
def channelState(array,ratio):
  [row,col]=np.shape(array)
  tmp=np.reshape(array,row*col)
  Max=np.max(tmp)
  Min=np.min(tmp)
  Thre=(Max-Min)*ratio
  state=np.zeros([row,col],bool)
  for i in range(0,row):
    for j in range(0,col):
      if array[i,j]-Min>Thre:
        state[i,j]=1
      else:
        state[i,j]=0
  return state

#calculate channel duration period
#state: state 0/1 array  low,high: index of channel in consider
def group_consecutives(vals, expect=0):
    """Return list of consecutive lists of numbers from vals (number list).""" 
    run = []    
    result = []#[run]
    for v in vals:
        if (v == expect):
            run.append(v)
        else:
            result.append(run)
            run = []
    return result
      

filepath = "D://Code//Matlab//SpectrumLearning//Data//AllTimeSlotinOne_1710_1740.mat"
data=scio.loadmat(filepath)
thisdata=data['AllTimeSlots']
tmpdat = statANA(thisdata,0,800)
Res=MaxMinNormalization(tmpdat)
Sta=channelState(Res,0.8)
val=np.array([0,1,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0])
resval=group_consecutives(val,0)

#
#newfilepath = "D://Code//WorkSpace//SpectrumPY//tmpdata.mat"
#tmpdat = statANA(thisdata,0,800)
#
#A=tmpdat
#[row,col]=np.shape(A)
#tmp=np.reshape(A,row*col)
#Max=np.max(tmp)
#Min=np.min(tmp)
#Diff=Max-Min
#X=np.zeros([row,col],float)
#for i in range(0,row):
#  for j in range(0,col):
#    #tmpa = A(i,j)
#    #X[i,j] += 10 #(tmpa-Min)/Diff
##Nomdat = MaxMinNormalization(tmpdat)
#print ("A[3,4]")
#scio.savemat(newfilepath,thisdata)