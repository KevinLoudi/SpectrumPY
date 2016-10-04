# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:21:42 2016

@author: kevin
"""
#use a api key
import quandl
import pandas as pd

#api_key=open('quandlapikey.txt','r').read()
#df=quandl.get('FMAC/HPI_AK',authtoken=api_key)
#load online data
df=quandl.get("FMAC/HPI_AK", authtoken="gyD-yf5K5vMXsuhsgV2_")
df.to_csv("LocalData.csv")
print(df.head())