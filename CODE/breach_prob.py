# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:53:40 2022

@author: gavgous
"""
# =============================================================================
# SCRIPT FOR BREACH PROBABILITY at LL 6.2.
# =============================================================================



import pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from thesis import breach_probability, CC
import numpy as np, seaborn as sns
from tqdm import tqdm
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.offline import plot
from statsmodels.distributions.empirical_distribution import ECDF
from UTILS import dcc_nll, create_dataset, create_dataset_pct
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)


path = r"C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Data"
data = pd.read_csv(path+"\\"+'87_ll_portf.csv', index_col = 'Date')
# data.index = pd.DatetimeIndex(data.index)

data = data/data.iloc[0]
data.plot()
plt.show()

dcc = CC(data, correlation_type = 'Dynamic', volatility = 'GARCH')
dcc.fit()

# dcc_lst = CC(data, correlation_type = 'Dynamic', volatility = 'LSTM')
# dcc_lst.fit()

res = breach_probability(dcc, start = 0, end = -1, liab_ratio = .9)
# breach_probability(dcc_lst, start = -0, end = -1, liab_ratio = .9)

# d = 5
# v_t_1 = data.shape[1]**(-1)*data.sum(1).values[d:]
# v_t = data.shape[1]**(-1)*data.sum(1).values[:-d]
# sns.histplot(v_t_1/v_t)

# np.quantile(v_t_1/v_t, q = .004)


# ===========================================================================
# for 6.2 (estimation of the lending value surface)
# ===========================================================================


simuls = dcc.DCC_GARCH_simulation(simulations = 1000, )
plt.plot(simuls.transpose().values)
plt.show()


closeouts = [5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60]
results = []
for closeout in tqdm(closeouts):
    
    total = create_dataset_pct(simuls, closeout = closeout).values.ravel()
    # sns.histplot(total, kde = True)
    # plt.show()
    
    ecdf = ECDF(total)
    
    y = sorted(set(total))
    x = [ecdf(y_) for y_ in y]
    inverted_edf = interp1d(x, y)
    
    epsilon = np.around(np.linspace(.001, .05), 4)

    # lam = lambda x:  np.exp(inverted_edf(x))
    lam = lambda x: inverted_edf(x)/100+1
    
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

