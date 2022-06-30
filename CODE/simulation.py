# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:51:19 2022

@author: Giorgio
"""

# =============================================================================
# SCRIPT FOR THE LL APPLICATION IN 6.3
# =============================================================================

from thesis import CC, DNN, RNN
from sklearn.metrics import mean_squared_error as mse
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl, tensorflow as tf
from tqdm import tqdm
from UTILS import dcc_nll,\
    take_DCC_cov, train_test_split_ts, forward_garch_multi, take_X_y,\
        cov_to_corr, simulate_garch, create_dataset, create_dataset_pct
from scipy.optimize import minimize
import plotly.graph_objects as go, seaborn as sns
from plotly.offline import plot
from arch import arch_model
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.graph_objects as go
from plotly.offline import plot
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)


path = r"C:\Users\gavgous\OneDrive - Zanders-BV\Desktop\THESIS\GITHUB\THESIS-main\DATA"
data = pd.read_csv(path+"\\"+'SMI_data.csv', 
                    index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)
data/=data.iloc[0]

fig, axs = plt.subplots(1,2)
data.plot(ax = axs[0])
sns.heatmap(data.pct_change().dropna().corr(), ax = axs[1], cmap='viridis')
plt.show()

train, test = train_test_split_ts(data)

dcc = CC(train, correlation_type = 'Dynamic', volatility = 'GARCH')
dcc.fit()
# dcc.visualize_loss_fn()


portf_val = dcc.DCC_GARCH_simulation(simulations = 1_000, horizon = 255)
portf_val.to_csv(path+'\\SimulDCCGARCH_3y_P1.csv')
plt.plot(portf_val.transpose())
plt.show()






# print('\nReading Simulations...')
# portf_val = pd.read_csv(
#     path+'\\SimulDCCGARCH_3y.csv',
#     index_col = ['Unnamed: 0'])
# print('\nSimulations Read!')

# plt.plot(portf_val.transpose().values)
# plt.plot()


closeouts = [5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60]
results = []
for closeout in tqdm(closeouts):
    
    total = create_dataset_pct(portf_val, closeout = closeout).values.ravel()
    # sns.histplot(total, kde = True)
    # plt.show()
    
    ecdf = ECDF(total)
    
    y = sorted(set(total))
    x = [ecdf(y_) for y_ in y]
    inverted_edf = interp1d(x, y)
    
    epsilon = np.around(np.linspace(.0001, .05), 4)

    # lam = lambda x:  np.exp(inverted_edf(x))
    lam = lambda x: inverted_edf(x)/100+1
    plt.plot(lam(epsilon))
    plt.show()
    
    results.append(lam(epsilon).tolist())

results = pd.DataFrame(results, index = closeouts, columns = epsilon)

results.transpose().plot()
plt.show()
plt.xlabel('ε')
plt.ylabel('λ')
plt.title('Lending Value λ for Different Closeout Periods')

fig = go.Figure(data=[go.Surface(colorscale='Viridis',
                                 z=results.values,
                                 x = epsilon,
                                 y = closeouts)])
fig.update_layout(
    scene = dict(
        xaxis = dict(
            title='ε'),
        yaxis = dict(
            title='closeout'),
        zaxis = dict(
            title='Lending Value Surface')
        )
    )
plot(fig)
        
sns.heatmap(results)
plt.show()

# #backtesting
# paths_test = pd.DataFrame(
#     np.sum(np.cumprod(1+test.pct_change().dropna().values,0),1)
#     ).transpose()


# v_t1 = np.cumprod(1+train.pct_change().dropna(),0).sum(1).iloc[30:].values
# v_t = np.cumprod(1+train.pct_change().dropna(),0).sum(1).iloc[:-30]

# sns.histplot(v_t1/v_t)

# v_t1 = np.cumprod(1+test.pct_change().dropna(),0).sum(1).iloc[10:].values
# v_t = np.cumprod(1+test.pct_change().dropna(),0).sum(1).iloc[:-10]

# sns.histplot(v_t1/v_t)

