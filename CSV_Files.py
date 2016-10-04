# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:40:06 2016

@author: kevin
"""

import pandas as pd

df = pd.read_csv('BCB-7469.csv')
#print(df.head())

#set Date as index
#df.set_index('Date', inplace = True)
#
##save back 
#df['Value'].to_csv('newcsv2.csv')
#
##read out
#df=pd.read_csv('newcsv2.csv')
#print(df.head())
#
#df=pd.read_csv('newcsv2.csv',index_col=0)
##modify column name
#df.columns=['Beijing_HPI']
#
##save data with on header
#df.to_csv('newcsv3.csv',header=False)
#
##add a header for the data read
#df = pd.read_csv('newcsv3.csv', names = ['Date','House_Price'], index_col=0)
#print(df.head())

#Convert File type
#convert csv to html
df.to_html('HTMLexample.html')
df = pd.read_csv('newcsv3.csv', names = ['Date','House_Price'], index_col=0)
print(df.head())
df.rename(columns={'House_Price':'Prices'}, inplace=True)
print(df.head())

