# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:20:08 2016

@author: kevin
"""

##create string
# print('Hello')
# print("Hello")
# print('''Hello''')
# 
# #
# print('Hello'[1])

import pandas as pd
import datetime
import pandas.io.data as web

#pulling data from Jan 1st 2010 to Aug 22nd 2015
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 8, 22)

#This pulls data for Exxon from the Yahoo Finance API
df = web.DataReader("XOM", "yahoo", start, end)

#print the dataframe for debugging
print(df)
print(df.head())

import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
#visualization
df['High'].plot()
plt.legend()
plt.show()