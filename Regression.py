# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:54:33 2017

@author: cen78179
"""
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def EstimateOLS(data,formula,showPlots = False):
    y,X = dmatrices(formula,data=data)
    ols = sm.OLS(y,X).fit(cov_type='HC3')
    print(ols.summary())

    if showPlots:
        plt.plot(ols.resid,'o')
        plt.title('OLS Residuals')
        plt.show()
        plt.hist(ols.resid,bins=20)
        plt.show()
        plt.figure(1)
        plt.plot(data['beta0'])
        data['OLSfitted'] = ols.fittedvalues
        plt.plot(data['beta0'])
        plt.plot(data['OLSfitted'])
        plt.show()
    
    return ols


def getFittedLevels(fitChange,firstObs,SeriesName):
    result = []
    lastObs = firstObs
    for change in fitChange:
        value = lastObs + change
        result.append(value)
        lastObs = value

    return pd.Series(result)

