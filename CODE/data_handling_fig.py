# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:29:21 2022

@author: gavgous
"""

import pandas as pd, yfinance as yf
import seaborn as sns, matplotlib.pyplot as plt, numpy as np
plt.rcParams['text.usetex'] = False

data = yf.Ticker('^GSPC').history(period = '3mo').Close

rets = 100*data.pct_change().dropna()


rv = np.log(rets.rolling(22).std())

fig, axs = plt.subplots(2,1, sharex = True)
axs[0].plot(rets.iloc[-23:-3], 'rs')
axs[0].plot(rets.iloc[-3:-2], 'bs')
axs[0].plot(rets.iloc[-40:-23], 'ks')
axs[0].axvline(x = rets.index[-4], c = 'grey', linestyle = '--')
axs[1].axvline(x = rets.index[-4], c = 'grey', linestyle = '--')
axs[0].axvline(x = rets.index[-23], c = 'grey', linestyle = '--')
axs[1].axvline(x = rets.index[-23], c = 'grey', linestyle = '--')
axs[1].plot(rv.iloc[-23:-3], 'rs')
axs[1].plot(rv.iloc[-40:-23], 'ks')
axs[0].set_title('Percentage Returns')
axs[1].set_title('Log-Realized-Volatility')
fig.legend(['Explanatory Data ', 'Target Data', 'Data Outside of the Rolling Window'],     # The line objects
           labels=['Explanatory Data ', 'Target Data', 'Data Outside of the Rolling Window'],   # The labels for each line
           loc="upper right",   # Position of legend
           borderaxespad=1,    # Small spacing around legend box
           )