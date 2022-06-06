# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:04:05 2022

@author: Giorgio
"""

import pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')


tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^SSMI', '^OEX', '^N225', '^FTSE']
start_date = '2000-01-01'
dataset = [yf.Ticker(t).history(start = start_date).Close\
           for t in tickers]

for i in range(len(dataset)):
    print(tickers[i])
    print(dataset[i].index[0], dataset[i].index[-1])
    print(dataset[i].shape)
    print(20*'-'+'\n\n')

returns = [100*d.pct_change().dropna() for d in dataset]

# RETURNS plot
fig, axs = plt.subplots(4, 2, squeeze=False, sharex = True, sharey = True)
for i in range(4):
    axs[i, 0].plot(returns[2*i])
    axs[i, 0].set_title(tickers[2*i][1:])
    axs[i, 1].plot(returns[2*i+1])
    axs[i,1].set_title(tickers[2*i+1][1:])
fig.suptitle('Returns Data', size = 30)
fig.tight_layout()
plt.show()

#AUTOCORRELATIONS plot
fig, axs = plt.subplots(4, 2, squeeze=False, sharex = True, sharey = True)
for i in range(4):
    pd.plotting.autocorrelation_plot(returns[2*i], ax = axs[i,0])
    axs[i, 0].set_title(tickers[2*i][1:])
    axs[i, 0].xaxis.set_label_text('')
    axs[i, 0].yaxis.set_label_text('')
    axs[i, 0].set_xlim([1, 30])
    pd.plotting.autocorrelation_plot(returns[2*i+1], ax = axs[i,1])
    axs[i,1].set_title(tickers[2*i+1][1:])
    axs[i, 1].xaxis.set_label_text('')
    axs[i, 1].yaxis.set_label_text('')
    axs[i, 1].set_xlim([1, 30])
fig.suptitle('Autocorrelations', size = 30)
fig.tight_layout()
for ax in axs.flat[-2:]:
    ax.set(xlabel='Lag', ylabel='')
for ax in axs.flat:
    ax.label_outer()
plt.show()

#HISTOGRAMMS plot
fig, axs = plt.subplots(4, 2, squeeze=False, sharex = True, sharey = True)
for i in range(4):
    sns.histplot(returns[2*i], bins = 200, kde = True, ax = axs[i,0], color = 'red')
    axs[i, 0].set_title(tickers[2*i][1:])
    axs[i, 0].xaxis.set_label_text('')
    axs[i, 0].yaxis.set_label_text('')
    sns.histplot(returns[2*i+1], bins = 200, kde = True, ax = axs[i,1], color = 'red')
    axs[i,1].set_title(tickers[2*i+1][1:])
    axs[i, 1].xaxis.set_label_text('')
    axs[i, 1].yaxis.set_label_text('')
fig.suptitle('Histogramms', size = 30)
fig.tight_layout()
for ax in axs.flat[-2:]:
    ax.set(xlabel='Lag', ylabel='')
for ax in axs.flat:
    ax.label_outer()
plt.show()

#Summary Statistics
statistics = pd.DataFrame([])
for i in range(len(tickers)):
    statistics = \
        pd.concat((statistics, returns[0].describe(percentiles = [.05, .5, .95])), axis=1)
statistics.columns = [t[1:] for t in tickers]
statistics.to_csv(r'C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Tables\sum_stats_1d.csv')
