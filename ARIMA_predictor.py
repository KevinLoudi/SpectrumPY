# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:38:59 2016
Propose: ARIMA in seansonal series prediction
Orign from http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
@author: kevin
Environment: Python 2.7
"""

#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
#import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
df.index.name=None
df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None

df.columns= ['riders']
df['riders'] = df.riders.apply(lambda x: int(x)*100)

df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)
plt.savefig('month_ridership.png', bbox_inches='tight')

#access each component of the decomposition
decomposition = seasonal_decompose(df.riders, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
trend = decomposition.trend
seasonal = decomposition.seasonal 
residual = decomposition.residual 

#we can acutally apply regression techniques to 
#stationary random variables
#law of large numbers and central limit theorem to name a couple

#two way to check stationarity 
#1)identify a changing mean or variation in the data
#2)Dickey-Fuller test for accurate assessment 

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
test_stationarity(df.riders)

df.riders_log= df.riders.apply(lambda x: np.log(x))  
test_stationarity(df.riders_log)

#first difference help to eliminate the overall trend
#improve the stationarity of the data
df['first_difference'] = df.riders - df.riders.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))

df['log_first_difference'] = df.riders_log - df.riders_log.shift(1)  
test_stationarity(df.log_first_difference.dropna(inplace=False))

#take a seasonal difference to remove the seasonality
df['seasonal_difference'] = df.riders - df.riders.shift(12)  
test_stationarity(df.seasonal_difference.dropna(inplace=False))

df['log_seasonal_difference'] = df.riders_log - df.riders_log.shift(12)  
test_stationarity(df.log_seasonal_difference.dropna(inplace=False))

#enhance seasonality by first difference of the seasonal difference
#p-value will be significantly reduced
df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(12)  
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))

df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(12)  
test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))

#ACF and PACF charts for the seasonal first difference
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[13:], lags=40, ax=ax2)

#make and the parameters for the model 
#mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(0,1,1,12))
#results = mod.fit()
#print results.summary()
#
#mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
#results = mod.fit()
#print results.summary()

#use the model to make prediction on the time series
df['forecast'] = results.predict(start = 102, end= 114, dynamic= True)  
df[['riders', 'forecast']].plot(figsize=(12, 8)) 
plt.savefig('ts_df_predict.png', bbox_inches='tight')

#compare the accaucy of prediction
npredict =df.riders['1982'].shape[0]
fig, ax = plt.subplots(figsize=(12,6))
npre = 12
ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'riders'], 'o', label='Observed')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')
plt.savefig('ts_predict_compare.png', bbox_inches='tight')

#generate new dataframe for future forcast
start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= df.columns)
df = pd.concat([df, future])

df['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  
df[['riders', 'forecast']].ix[-24:].plot(figsize=(12, 8)) 
plt.savefig('ts_predict_future.png', bbox_inches='tight')

## A little extra I was doing to try and include an exogenous variable for number of weekdays in each month.
## Thinking that people take public transportation more during the week, a count of weekdays in each month could help explain some of the variance.  

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
moving = start
d = {}
year =0
month =0
while moving < datetime.datetime(1982,7,1):
#     print moving
    if moving.year == year:
        if moving.month == month:
            if moving.weekday() < 5:
                d[str(moving.year)+"-"+ str(moving.month)] += 1
        else:
            d[str(moving.year)+"-"+ str(moving.month)]=0
            if moving.weekday() < 5:
                d[str(moving.year)+"-"+ str(moving.month)] += 1
    else:
#         d[moving.year] = {}
        d[str(moving.year)+"-"+ str(moving.month)]=0
        if moving.weekday() < 5:
            d[str(moving.year)+"-"+ str(moving.month)] += 1


    year = moving.year
    month = moving.month
    moving += datetime.timedelta(days=1)
df_dow = pd.DataFrame(d.items(), columns=['Month', 'DateValue'])
df_dow.Month = pd.to_datetime(df_dow.Month)
df_dow.sort('Month', inplace=True)

def holiday_adj(x):
    if x['Month'].month==1:
        x['DateValue'] -=1
        return x['DateValue'] 
    elif x['Month'].month==2:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==5:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==7:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==9:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==10:
        x['DateValue'] -=1
        return x['DateValue']    
    elif x['Month'].month==11:
        x['DateValue'] -=3   
        return x['DateValue']
    elif x['Month'].month==12:
        x['DateValue'] -=2
        return x['DateValue']

    else:
        return x['DateValue']
    
df_dow['days'] = df_dow.apply(holiday_adj, axis=1)
df_dow.set_index('Month', inplace=True)
df_dow.index.name = None