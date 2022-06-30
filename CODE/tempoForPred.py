# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:47:17 2022

@author: gavgous
"""

# =============================================================================
# temporary for prediction
# =============================================================================

from thesis import CC
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl
from tqdm import tqdm
from UTILS import dcc_nll,\
    take_DCC_cov, train_test_split_ts, forward_garch_multi, take_X_y
from arch import arch_model
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)

start = time.time()
path = r"C:\Users\gavgous\OneDrive - Zanders-BV\Desktop\THESIS\GITHUB\THESIS-main\DATA"

# =============================================================================
# available data sets:
#    - FTSE_dat.csv
#    - SMI_data.csv
#    - Berkshire_Hathaway_Portfolio_data.csv
# =============================================================================

datasets = \
    ['FTSE_data.csv',
      'SMI_data.csv',
      'Berkshire_Hathaway_Portfolio_data.csv']

for dataset in datasets:
    data = pd.read_csv(path+"\\"+dataset, index_col = 'Date')
    data.index = pd.DatetimeIndex(data.index)
    
    data = data/data.iloc[0]

    dcc_lstm = CC(data, correlation_type = 'Dynamic', volatility = 'LSTM')
    dcc_lstm.fit()
    
    dcc_garch = CC(data, correlation_type = 'Dynamic', volatility = 'GARCH')
    dcc_garch.fit()
    
    cov_garch = dcc_garch.forecast_covariance(5)
    cov_lstm = dcc_lstm.forecast_covariance(5)
    
    
    plt.plot(cov_garch.sum(1).sum(1), label = 'garch')
    plt.plot(cov_lstm.sum(1).sum(1), label = 'lstm')
    plt.legend()
    plt.title('Forecast on '+dataset)
    plt.show()


#TODO: extend to GARCH
#TODO: functionalize
#TODO: breach probability