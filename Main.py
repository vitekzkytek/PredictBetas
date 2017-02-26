# -*- coding: utf-8 -*-
import TimeSeries
import Regression

import datetime
import pandas as pd
import numpy as np
from patsy import dmatrices,dmatrix
import statsmodels.api as sm
#import statsmodels.tsa.stattools.adfuller as adfuller
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt

FI_CALC_Path = "betas_input.xlsx"

def pdLoadBetas(start,end):
    exdf = pd.read_excel(FI_CALC_Path,'input', header=0,index_col=0)
    df = pd.DataFrame()
    for i,s in exdf.iterrows():
        if i.to_datetime().date() >= start and i.to_datetime().date() <= end:
            df = df.append(s)
    return df


start = datetime.date(2004,1,1)
end = datetime.date(2017,2,1)
data = pdLoadBetas(start,end)
shortData = data['2011-02-01':]
#TIME SERIES ANALYSIS
if False:
    d = data['beta0'] 
    TimeSeries.test_stationarity(d,'Original') #beta0 NonStationary
    diffs = d.diff()[1:len(d)] 
    TimeSeries.test_stationarity(diffs,'Diffs') #diffs stationary
    TimeSeries.ACF_analysis(diffs)
    
    logdiffs = np.log(d).diff()[1:len(d)]
    TimeSeries.test_stationarity(logdiffs,'Logdiffs')
    TimeSeries.ACF_analysis(logdiffs)
    
    TimeSeries.ARIMA_auto_fit(d)
    #BIC: DIFFs - (1,1,0), LogDiffs - (0,1,1)
    #AIC: DIFFs - (5,1,1), LogDiffs - (4,1,1)

    breakpoint = '2017-02-01'
    mod = TimeSeries.AR_predict(d[:breakpoint])
    
    
    plt.plot(d)
    plt.plot(mod.predict(start=breakpoint,end='2017-12-01'))
    print(mod.params)
    
    
#REGRESSIONS
if False:
    plt.figure(1)
    plt.subplot(221)
    plt.plot(shortData['beta0'])
    plt.title('Beta 0')
    plt.subplot(222)
    plt.plot(shortData['CNBrepo'])
    plt.title('CNB repo rate')
    plt.subplot(223)
    plt.plot(shortData['y10Yforecast'])
    plt.title('10Y Yield forecast')
    plt.subplot(224)
    plt.plot(shortData['PriborSpread'])
    plt.title('Implied forward 1Y')
    plt.show()
     
    ols = Regression.EstimateOLS(shortData,'beta0 ~ y10Yforecast + ImpFwd1Y + CNBrepo')



if True:
    diffsdata = data['2011-02-01':].diff()[1:len(data)]
    #print(diffsdata)
    ols = Regression.EstimateOLS(diffsdata, 'beta0 ~ y10Yforecast + CNBrepo',False)
    #print(ols.summary())
    #plt.figure(1)
    #plt.plot(diffsdata['beta0'])
    #plt.plot(ols.fittedvalues)
    #plt.show()
    shortData['b0_fitted'] = Regression.getFittedLevels(ols.fittedvalues,shortData.ix[0,'beta0'],'FittedBeta0')
    plt.figure(1)
    plt.plot(shortData['beta0'])
    plt.plot(shortData['b0_fitted'])
    plt.show()
    pass
    #plt.plot(ols.fittedvalues)
    #plt.show()




#plt.plot(mod.predict(start='2017-10-01',end='2017-12-01'))

#
#
#adf = adfuller(d)
#plt.plot(d)
#
#plt.plot(d.diff())
#
##sm.graphics.tsa.plot_acf(d.diff())
##sm.graphics.tsa.plot_pacf(d.diff())
#
#sm.graphics.tsa.plot_acf(d)
#sm.graphics.tsa.plot_pacf(d)
#
#
#ar = tsa.ar_model.AR(d.diff()[1:len(d.diff())]).fit()
##ar.fit()
#
#


#Y,X = dmatrices('beta0 ~ CNBrepo + CPI + EURCZK + EURIBOR + PriborSpread',data=data)

#ols = sm.OLS(Y,X).fit()
#print(ols.summary())

#
#plt.figure(1)
#plt.subplot(321)
#plt.plot(data['PriborSpread'])
#plt.title('PriborSpread')
#plt.subplot(322)
#plt.plot(data['EURIBOR'])
#plt.title('EURIBOR')
#plt.subplot(323)
#plt.plot(data['CNBrepo'])
#plt.title('CNBrepo')
#plt.subplot(324)
#plt.plot(data['EURCZK'])
#plt.title('EURCZK')
#plt.subplot(325)
#plt.plot(data['CPI'])
#plt.title('CPI')
#plt.subplot(326)
#plt.plot(data['beta0'])
#plt.title('beta0')
#plt.show()
#
#
#plt.plot(ols.resid,'o')
#plt.title('OLS Residuals')
#plt.plot(data['beta0'])
#print(data)


#beta0 = np.arrays()