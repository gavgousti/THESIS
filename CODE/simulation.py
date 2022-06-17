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
    take_DCC_cov, train_test_split_ts, forward_garch_multi, take_X_y,\
        cov_to_corr, simulate_garch, create_dataset
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
data = pd.read_csv(path+"\\"+'Berkshire_Hathaway_Portfolio_data.csv', index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

dcc = CC(data, correlation_type = 'Dynamic', volatility = 'GARCH')
dcc.fit()



portf_val = dcc.DCC_GARCH_simulation(simulations = 1_000)
plt.plot(portf_val.transpose())
plt.plot()


closeouts = [5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60]
results = []
for closeout in closeouts:
    
    total = create_dataset(portf_val, closeout = closeout).values.ravel()
    
    
    ecdf = ECDF(total)
    
    y = sorted(set(total))
    x = [ecdf(y_) for y_ in y]
    inverted_edf = interp1d(x, y)
    
    epsilon = np.around(np.linspace(.0001, .1), 4)

    lam = lambda x:  np.exp(inverted_edf(x))
    
    results.append(lam(epsilon).tolist())

results = pd.DataFrame(results, index = closeouts, columns = epsilon)

results.transpose().plot()
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
# for i in range(len(dcc.stock_names)):
#     plt.plot(simulate_garch(dcc.garch_models[dcc.stock_names[i]].params,
#                                       simulations = 1,
#                                       horizon =255)['Volatilities'].values.ravel())