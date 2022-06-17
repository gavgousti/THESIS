# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:51:19 2022

@author: Giorgio
"""
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

def simulate_garch(
        params,
        simulations = 100,
        horizon = 10
        ):
    omega, alpha, beta = params.omega, params['alpha[1]'], params['beta[1]']
    output = []
    for _ in tqdm(range(simulations)):
        vola_path = [np.sqrt(omega/(1-alpha-beta))]
        r = np.random.randn()*vola_path[-1]
        for _ in range(horizon):
            vola_path.append(np.sqrt(omega\
                                     +alpha*(r/vola_path[-1])**2\
                                         +beta*vola_path[-1]**2))
            r = np.random.randn()*vola_path[-1]
        output.append(vola_path)
    return pd.DataFrame(output, columns=['h'+str(i) for i in range(horizon+1)])

path = r"C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Data"
data = pd.read_csv(path+"\\"+'SMI_data.csv', index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

dcc = CC(data, correlation_type = 'Dynamic', volatility = 'GARCH')
dcc.fit()


simulate_garch(dcc.garch_models[dcc.stock_names[0]].params,
                                     simulations = 3,
                                     horizon =255)

def simulate_dcc(
        model,
        simulations = 10, 
        horizon = 255
        ):
    for _ in tqdm(range(simulations)):
        
        
            