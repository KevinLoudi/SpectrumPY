# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:00:42 2016

@author: kevin
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

#use dictories
web_stats={'Day':[1,2,3,4,5,6],
           'Vistors':[43,53,34,56,34,89],
           'Bounce Rate':[65,67,78,65,45,52]}

df=pd.DataFrame(web_stats)
print(df.Vistors)
print(df['Vistors'].tolist())
print(np.array(df[['Vistors','Bounce Rate']]))

df2 = df.set_index('Day', inplace=False)
print(df2)

df.plot()
plt.show()
#print(df)
#print(df.head())
#print(df.tail(2))

#returned a new data frame
#df2=df.set_index('Day')

#df2=df.set_index('Day',inplace=True)
#print(df2)

