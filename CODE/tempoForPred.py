# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:47:17 2022

@author: gavgous
"""

# =============================================================================
# temporary for prediction
# =============================================================================

import lightgbm as lgb
from thesis import CC, DNN, RNN, GB
from sklearn.metrics import mean_squared_error as mse
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl, tensorflow as tf
from tqdm import tqdm
from UTILS import dcc_nll,\
    take_DCC_cov, train_test_split_ts, forward_garch_multi, take_X_y
from scipy.optimize import minimize
import plotly.graph_objects as go, seaborn as sns
from plotly.offline import plot
import datetime
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
    
    train, test = train_test_split_ts(data)
    
    vola_types = ['GARCH', 'EGARCH', 'GJR', 'GB', 'FNN', 'LSTM']

    RES = []
    for vola in vola_types:
        # dcc = CC(train, correlation_type = 'Dynamic', volatility = vola)
        # dcc.fit()
        # RES.append(dcc.evaluation_on_test_set(test))
        

        # ccc = CC(train, correlation_type='Constant', volatility = vola)
        # ccc.fit()
        break
    break

# =============================================================================
# 1d prediction
# =============================================================================

# =============================================================================
# for RNN
# =============================================================================
lag = 20; include_rv = True; lstm = True
rets = 100*data.iloc[:,0].pct_change().dropna()

# =============================================================================
# RNN
# =============================================================================

rets_train, rets_test = train_test_split_ts(rets, .7)
X_train, y_train = take_X_y(rets_train, lag, reshape = False, take_rv = include_rv, log_rv =include_rv )
X_test, y_test = take_X_y(rets_test, lag, reshape = False, take_rv = include_rv, log_rv =include_rv)

X_train, y_train = tf.convert_to_tensor(X_train, dtype  = 'float32'),\
                        tf.convert_to_tensor(y_train, dtype  = 'float32') 

X_test, y_test = tf.convert_to_tensor(X_test, dtype  = 'float32'),\
                        tf.convert_to_tensor(y_test, dtype  = 'float32') 

rnn = RNN(
    lstm = lstm,
    hidden_size = [60],
    hidden_activation = 'tanh',
    last_activation = 'exponential',
    dropout = 0.0,
    l1 = 5,
    l2 = 0
)

rnn.train(
    X_train, 
    y_train,
    X_test,
    y_test,
    epochs = 10,
    bs = 2048,
    lr = .008
)
prediction_rnn = rnn.predict_1d(X_train = X_train, horizon = 5)

# =============================================================================
# FNN
# =============================================================================

X_train_dnn, y_train_dnn = take_X_y(rets_train, lag, reshape = True, take_rv = include_rv, log_rv =include_rv )
X_test_dnn, y_test_dnn = take_X_y(rets_test, lag, reshape = True, take_rv = include_rv, log_rv =include_rv)

X_train_dnn, y_train_dnn = tf.convert_to_tensor(X_train_dnn, dtype  = 'float32'),\
                        tf.convert_to_tensor(y_train_dnn, dtype  = 'float32') 

X_test_dnn, y_test_dnn = tf.convert_to_tensor(X_test_dnn, dtype  = 'float32'),\
                            tf.convert_to_tensor(y_test_dnn, dtype  = 'float32') 
dnn = DNN(
    hidden_size = [300],
    dropout = .5,
    l1 = 1,
    l2 = 1
)

dnn.train(
    X_train_dnn, 
    y_train_dnn,
    X_test_dnn,
    y_test_dnn,
    epochs = 20,
    bs = 1024,
    lr = .001,
)
prediction_dnn = dnn.predict_1d(X_train = X_train_dnn, horizon = 5)

# =============================================================================
# GB
# =============================================================================

X_train_gb, y_train_gb = take_X_y(rets_train, lag, take_rv = include_rv, log_rv =include_rv, reshape = True )
X_test_gb, y_test_gb = take_X_y(rets_test, lag, take_rv = include_rv, log_rv =include_rv, reshape = True)
lgb_train_gb, lgb_test_gb = lgb.Dataset(X_train_gb, y_train_gb, free_raw_data=False ),\
lgb.Dataset(X_test_gb, y_test_gb,  free_raw_data=False )

gb = GB()
gb.fit(rets_train)
plt.plot(gb.predict(rets_train.iloc[-50:]))
