# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:44:23 2016
Propose: annual sunspot data from 1700 – 2008 
recording the number of sunspots per year with ARMA
Origined from http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html
@author: kevin
"""
import numpy as np
from scipy import stats
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

#need connection to Internet
print sm.datasets.sunspots.NOTE
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pandas.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
dta.plot(figsize=(12,8)); #??

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
#plot ACF (autocorrelation function) and PACF (partial autocorrelation)
#correlation between singal self and other time (difference between tow observation)
#mean to find the repeated pattern

#Seasonal patterns of time series can be examined via correlograms
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
# examine serial dependencies
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

#The value of Durbin-Watson statistic is close to 2 if the errors 
#are uncorrelated. In our example, it is 0.1395. That means that 
#there is a strong evidence that the variable open has high autocorrelation.
print sm.stats.durbin_watson(dta)

# show plots in the notebook
#%matplotlib inline
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

from pandas.tools.plotting import autocorrelation_plot
# show plots in the notebook
#%matplotlib inline
dta['SUNACTIVITY_2'] = dta['SUNACTIVITY']
dta['SUNACTIVITY_2'] = (dta['SUNACTIVITY_2'] - dta['SUNACTIVITY_2'].mean()) / (dta['SUNACTIVITY_2'].std())
plt.acorr(dta['SUNACTIVITY_2'],maxlags = len(dta['SUNACTIVITY_2']) -1, linestyle = "solid", usevlines = False, marker='')
#Autocorrelation function is a mixture of exponentials and 
#damped sine waves after (q-p) lags
plt.show()
autocorrelation_plot(dta['SUNACTIVITY'])
plt.show()

#Modeling time series
#first subtract sample-mean reduced to zero-mean

arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print arma_mod20.params
