# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:00:16 2022

@author: gavgous
"""

# =============================================================================
# script for vola figures
# =============================================================================

from tqdm import tqdm
from thesis import deployment_GB_1d, deployment_RNN_1d, deployment_DNN_1d
import matplotlib.pyplot as plt, matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm
import pylab as py
import numpy as np
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)
tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^SSMI', '^OEX', '^N225', '^FTSE']

res = []
for t in tqdm(tickers):
    gb = deployment_GB_1d(index = t, save = None)
    rnn = deployment_RNN_1d(lstm = False, index = t, save = None)
    lstm = deployment_RNN_1d(lstm = True, index = t, save = None)   
    fnn = deployment_DNN_1d(index = t, save = None)

    res.append([
        (rnn['sigma-r'][0]).numpy().ravel(),
        (lstm['sigma-r'][0]).numpy().ravel(),
        (fnn['sigma-r'][0]).numpy().ravel(),
        np.exp(gb['GARCH sigma-r'][0])**.5,
        np.exp(gb['GARCH sigma-r'][0])**.5,
        np.exp(gb['GARCH sigma-r'][0])**.5,
        np.exp(gb['GARCH sigma-r'][0])**.5,
        gb['RV']
        ]
        )
    
models = ['RNN', 'LSTM', 'FNN', 'GB', 'GARCH', 'GJR', 'EGARCH', 'RV']
fig, axs = plt.subplots(4, 2, sharex=False, sharey = False, squeeze=True, figsize = (13,15))    
for index in range(len(res)):
    axs.flat[index].set_title(tickers[index][1:], size = 10)
    for model in range(len(res[index])):
        if model!=len(res[index])-1:
            sns.lineplot(x = np.arange(0,res[index][model].shape[0]),
                         y = res[index][model],
                         alpha = .8,
                         ax = axs.flat[index],
                         # label=models[model],
                         linewidth = .5)
        else:
            sns.lineplot(x = np.arange(0,res[index][model].shape[0]),
                         y = res[index][model],
                         alpha = 1,
                         ax = axs.flat[index],
                         # label = models[model],
                         color = 'red')
fig.legend(tickers,     # The line objects
           labels=models,   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )
