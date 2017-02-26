# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:19:54 2017

@author: cen78179
"""
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
#from statsmodels.graphics.tsa import plot_acf,plot_pacf
#import statsmodels.tsa.stattools.adfuller as adfuller
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np 

def test_stationarity(ts,TS_title):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(TS_title + ': Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print(TS_title +': Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    
    
def ACF_analysis(ts):
    sm.graphics.tsa.plot_acf(ts)
    sm.graphics.tsa.plot_pacf(ts)
    
    
def ARIMA_auto_fit(ts):
    #First differences
    diff = ts.diff()[1:len(ts)]
    kwazi = {}
    kwazi['warn_convergence'] = False
    res = arma_order_select_ic(diff,max_ar=6,max_ma=6,ic='aic',fit_kw=kwazi)
    print('First Differences')
    print(res)
    
    logdiff = np.log(ts).diff()[1:len(ts)]
    res = arma_order_select_ic(logdiff,max_ar=6,max_ma=6,ic='aic',fit_kw=kwazi)
    print('LogDifferences')
    print(res)
    pass


def AR_predict(ts):
    return sm.tsa.AR(ts).fit(maxlag=1,method='cmle')
    plt.plot(d)
    plt.plot(mod.fittedvalues)
    plt.show()
    
    #print(model)