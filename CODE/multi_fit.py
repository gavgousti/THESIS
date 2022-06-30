# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:32:30 2022

@author: gavgous
"""

# =============================================================================
#Fitting Multivariate models
# =============================================================================

#TODO: do it function

from thesis import CC, DNN, RNN
from sklearn.metrics import mean_squared_error as mse
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl, tensorflow as tf
from tqdm import tqdm
from UTILS import dcc_nll,\
    take_DCC_cov, train_test_split_ts, forward_garch_multi, take_X_y
from scipy.optimize import minimize
import plotly.graph_objects as go, seaborn as sns
from plotly.offline import plot
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
all_times = []
for dataset in datasets:
    data = pd.read_csv(path+"\\"+dataset, index_col = 'Date')
    data.index = pd.DatetimeIndex(data.index)
    data/=data.iloc[0]

    
    train, test = train_test_split_ts(data)
    
    vola_types = ['GARCH', 'EGARCH', 'GJR', 'GB', 'FNN', 'LSTM']
    times = []
    RES = []
    for vola in vola_types:
        dcc = CC(train, correlation_type = 'Dynamic', volatility = vola)
        dcc_time = -time.time()
        dcc.fit()
        dcc_time += time.time()
        RES.append(dcc.evaluation_on_test_set(test))
        plt.plot(len(dcc.stock_names)**(-1)*dcc.covariances_test.sum(1).sum(1)[22:]**.5,
                  label = 'DCC-'+vola, alpha = .8)


        ccc = CC(train, correlation_type='Constant', volatility = vola)
        ccc_time = -time.time()
        ccc.fit()
        ccc_time += time.time()
        RES.append(ccc.evaluation_on_test_set(test))
        plt.plot(len(dcc.stock_names)**(-1)*ccc.covariances_test.sum(1).sum(1)[22:]**.5,
                  label = 'CCC-'+vola, alpha = .8)
        times.append([dcc_time, ccc_time])
    plt.plot(len(dcc.stock_names)**(-1)*dcc.real_covariance_test.sum(1).sum(1)**.5,
              label = 'RPV', alpha = 1, color = 'red')
    plt.legend()
    plt.show()
    table = []
    all_times.append(times)
    for i in range(len(RES)):
        table.append([RES[i][key] for key in RES[i].keys()])
    table = pd.DataFrame(table)
    table.columns = RES[0].keys()
    table.set_index('MODEL', inplace = True)
    table.to_csv(
        'C:\\Users\\gavgous\\OneDrive - Zanders-BV\Desktop\\THESIS\GITHUB\\THESIS-main\Tables\\table_'\
            +dataset
            )
print("ELAPSED IN {:4.0f}'".format(-start+time.time()))
