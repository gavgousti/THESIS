# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:32:30 2022

@author: gavgous
"""

# =============================================================================
# TEMPORARAY SCRIPT
# =============================================================================


from thesis import CCC
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl
from tqdm import tqdm
from UTILS import take_pairs, nll_md, dcc_comp_nll
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.offline import plot
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)

path = 'C:\\Users\\gavgous\\OneDrive - Zanders-BV\\Desktop\\THESIS\\CODE\\THESIS-main\\'
data = pd.read_csv(path+"SMI_data.csv", index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

ccc = CCC(data, scale = 100)
ccc.fit()

sns.heatmap(ccc.P_c, cmap = 'viridis')
plt.title('Constant Correlation')
plt.show()

ccc.plot_cov_feature('det')
ccc.plot_cov_feature('trace')
ccc.plot_cov_feature('sum')

# =============================================================================
# test what explodes
# =============================================================================

ccc.plot_cov_feature('det')
#the determinant explodes

prod = []
for t in range(ccc.returns.shape[0]):
    Sigma_t = ccc.cov[t, :, :]
    x_ = ccc.returns.iloc[t].values.reshape(-1,1)
    prod.append((np.transpose(x_)@np.linalg.inv(Sigma_t)@x_)[0][0])
plt.plot(ccc.returns.index, prod)
plt.show()
# the product is quite stable

det3 = []
for t in range(ccc.returns.shape[0]):
    Sigma_t = ccc.cov[t, :, :]
    x_ = ccc.returns.iloc[t].values.reshape(-1,1)
    det3.append(np.log(np.linalg.det(Sigma_t)))
plt.plot(ccc.returns.index, det3)
plt.show()

plt.plot(ccc.returns.index, np.array(prod)+np.array(det3))
plt.show()

plt.plot(ccc.returns.index, .5*np.cumsum(np.array(prod)+np.array(det3)))
plt.show()

#Explosion seems to not matter at all as log fixes it

# =============================================================================
# -----------------------------------------------------------------------------
# =============================================================================




# =============================================================================
# OPTIMIZATION 
# =============================================================================

def constraint(theta):
    return 1-theta[0]-theta[1]

res = minimize(
    fun =  dcc_comp_nll,
    x0 = (.001, .001),
    args = (ccc.garch_volatilities, ccc.returns, 'full', 1),
    method = 'SLSQP',
    bounds = [(0,1), (0,1)],
    constraints = {'type': 'ineq', 'fun': constraint},
    options = {'maxiter': 20, 'disp': True}    
    )

res

theta = res.x

def take_DCC_cov(theta, sigma, P_c, x):
    P_c = P_c.values
    prev_corr = P_c.copy()
    alpha, beta = theta
    cov = np.full((x.shape[0], x.shape[1], x.shape[1]), .0)
    for t in range(x.shape[0]):
        x_ = x.iloc[t].values.reshape(-1,1)
        sigma_ = sigma.iloc[t].values    
        
        Sigma_t = np.diag(sigma_)@\
            ((1-alpha-beta)*P_c+alpha*x_@np.transpose(x_)+beta*prev_corr)@\
                np.diag(sigma_)
        cov[t, :, :] = Sigma_t
        prev_corr = np.diag(1/sigma_)@Sigma_t@np.diag(1/sigma_)
    return cov
    
dcc_cov = take_DCC_cov(theta, ccc.garch_volatilities, ccc.P_c, ccc.returns)

for t in tqdm(range(0, ccc.returns.shape[0], 60)):
    sns.heatmap(dcc_cov[t, :, :])
    plt.title(ccc.returns.index[t])
    plt.show()

plt.plot(np.sum(np.sum(dcc_cov, axis = 1 ), axis=1))

# =============================================================================
# try a different optimization approach
# =============================================================================
N = 15
values = []
alpha_grid = np.linspace(0, .2, N)
beta_grid = np.linspace(0, .2, N)
for alpha in tqdm(alpha_grid):
    tempo = []
    for beta in beta_grid:
        if beta+alpha<.9:
            tempo.append(dcc_comp_nll((alpha, beta),
                                      ccc.garch_volatilities,
                                      ccc.returns,
                                      'full',
                                      .1)
                         )
        else:
            tempo.append(np.nan)
    values.append(tempo)
values = pd.DataFrame(values, columns = beta_grid, index = alpha_grid)

sns.heatmap(values, vmax=None)
plt.show()

fig = go.Figure(data=[go.Surface(z=values.values, x = beta_grid, y = alpha_grid)])
fig.update_layout(
    xaxis_title="beta",
    yaxis_title="alpha"
)
plot(fig)

values.argmin()

theta = (.071, .071)
dcc_cov = take_DCC_cov(theta, ccc.garch_volatilities, ccc.P_c, ccc.returns)

for t in tqdm(range(0, ccc.returns.shape[0], 60)):
    sns.heatmap(dcc_cov[t, :, :])
    plt.title(ccc.returns.index[t])
    plt.show()

plt.plot(np.sum(np.sum(dcc_cov, axis = 1 ), axis=1))

dcc_comp_nll(theta, ccc.garch_volatilities, ccc.returns, 'full',1)
dcc_comp_nll((0,0), ccc.garch_volatilities, ccc.returns, 'full',1)

# =============================================================================
# TO SUM UP, IT SEEMS THAT THE MINIMUM DOES NOT NECESSARILY FINDS A 
# NICE ESTIMATOR. WE HAVE TO HAVE A LOOK AGAIN IN THE LOSS FUNCTIONS
# AND SEE IF WE CAN SOMEHOW MAKE THE OPTIMIZATION MORE STABLE !!! (17/05/22)
# =============================================================================
