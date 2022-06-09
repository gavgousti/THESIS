# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:40:12 2022

@author: gavgous
"""

# =============================================================================
# Script for Obtaining the Residuals
# =============================================================================

from tqdm import tqdm
from thesis import deployment_GB_1d, deployment_RNN_1d, deployment_DNN_1d
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pylab as py
import numpy as np
tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^SSMI', '^OEX', '^N225', '^FTSE']

res = []
for t in tqdm(tickers):
    gb = deployment_GB_1d(index = t, save = None)
    rnn = deployment_RNN_1d(lstm = False, index = t, save = None)
    lstm = deployment_RNN_1d(lstm = True, index = t, save = None)   
    fnn = deployment_DNN_1d(index = t, save = None)

    res.append([
        (rnn['sigma-r'][1]/rnn['sigma-r'][0]).numpy().ravel(),
        (lstm['sigma-r'][1]/lstm['sigma-r'][0]).numpy().ravel(),
        (fnn['sigma-r'][1]/fnn['sigma-r'][0]).numpy().ravel(),
        gb['sigma-r'][1]/np.exp(gb['GARCH sigma-r'][0])**.5,
        gb['GARCH sigma-r'][1]/np.exp(gb['GARCH sigma-r'][0])**.5,
        gb['EGARCH sigma-r'][1]/np.exp(gb['GARCH sigma-r'][0])**.5,
        gb['GJR sigma-r'][1]/np.exp(gb['GARCH sigma-r'][0])**.5
        ]
        )


def QQ(res, ax = None, N = 1000):
    y = np.quantile(res, np.linspace(0,1,N))
    x = np.quantile(np.random.randn(N), np.linspace(0,1,N))
    sns.scatterplot(x, y, alpha = 1, ax = ax)
    sns.lineplot(np.linspace(-5, 5), np.linspace(-5, 5), alpha = .5, ax = ax)


models = ['RNN', 'LSTM', 'FNN', 'GB', 'GARCH', 'GJR', 'EGARCH']
fig, axs = plt.subplots(7,len(tickers), sharex=True, sharey = True, squeeze=True, figsize = (13,15) )
for i in range(len(res[0])):
    for j in range(len(res)):
        QQ(res[j][i], axs[i, j])
        axs[i, j].set_xlim([-5, 5])
        axs[i, j].set_xlim([-5, 5])
        axs[i, j].xaxis.set_label_text(tickers[j][1:])
        axs[i, j].yaxis.set_label_text(models[i])

