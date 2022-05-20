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
from UTILS import dcc_nll_fixed, pseudo_dcc_nll
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.offline import plot
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)

path = "C:\\Users\\Giorgio\\Desktop\\Master\\THESIS CODES ETC\\Data\\"
data = pd.read_csv(path+"SMI_data.csv", index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

ccc = CCC(data, scale = 100)
ccc.fit()
print(ccc.nll)

sns.heatmap(ccc.P_c, cmap = 'viridis')
plt.title('Constant Correlation')
plt.show()

ccc.plot_cov_feature('det')
ccc.plot_cov_feature('trace')
ccc.plot_cov_feature('sum')



# =============================================================================
# OPTIMIZATION 
# =============================================================================
def take_DCC_cov(theta, sigma, P_c, x):
    cov = np.full((x.shape[0], x.shape[1], x.shape[1]), .0)
    alpha, beta = theta
    Q_bar = np.divide(x, sigma).cov().values
    Q_t = Q_bar.copy()
    nll = 0
    for t in tqdm(range(x.shape[0])):
        x_ = x.iloc[t].values.reshape(-1,1)
        sigma_ = sigma.iloc[t].values.reshape(-1,1)
        eps = np.divide(x_, sigma_)
        
        Q_t = (1-alpha-beta)*Q_bar+\
            alpha*eps@eps.transpose()+\
                beta*Q_t
        
        P_t = np.diag(np.diag(1/np.sqrt(Q_t)))@\
            Q_t@\
                np.diag(np.diag(1/np.sqrt(Q_t)))
                
        Sigma_t = np.diag(sigma_.ravel())@P_t@np.diag(sigma_.ravel())
                
        cov[t, :, :] = Sigma_t
    return cov



res = minimize(
    fun =  dcc_nll_fixed,
    x0 = (.05, .05),
    args = (ccc.garch_volatilities, ccc.returns),
    method = 'SLSQP',
    bounds = [(0,1), (0,1)],
    constraints = {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]},
    options = {'maxiter': 30, 'disp': True, 'full_output':True}    
    )

res

res_pseudo = minimize(
    fun =  pseudo_dcc_nll,
    x0 = (.05, .05),
    args = (ccc.garch_volatilities, ccc.returns),
    method = 'SLSQP',
    bounds = [(0,1), (0,1)],
    constraints = {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]},
    options = {'maxiter': 30, 'disp': True}    
    )

res_pseudo

theta_pseudo = res_pseudo.x
theta = res.x
    

dcc_cov = take_DCC_cov(theta, ccc.garch_volatilities, ccc.P_c, ccc.returns)

plt.plot(np.sum(np.sum(dcc_cov, axis = 1 ), axis=1), label = 'DCC')
plt.plot(np.sum(np.sum(take_DCC_cov((0,0),
                                    ccc.garch_volatilities,
                                    ccc.P_c,
                                    ccc.returns), axis = 1 ), axis=1),
         label = 'CCC'
         )
plt.legend()
plt.show()

print('\nCCC NLL: {}'.\
      format(dcc_nll_fixed((0,0), ccc.garch_volatilities, ccc.returns)))
print('\nDCC NLL: {}'.\
      format(dcc_nll_fixed(theta, ccc.garch_volatilities, ccc.returns)))
    
# =============================================================================
# TO SUM UP, IT SEEMS THAT THE MINIMUM DOES NOT NECESSARILY FINDS A 
# NICE ESTIMATOR. WE HAVE TO HAVE A LOOK AGAIN IN THE LOSS FUNCTIONS
# AND SEE IF WE CAN SOMEHOW MAKE THE OPTIMIZATION MORE STABLE !!! (17/05/22)
# =============================================================================

N = 25
values = []
alpha_grid = np.linspace(0, .999, N)
beta_grid = np.linspace(0, .99, N)
for alpha in tqdm(alpha_grid):
    tempo = []
    for beta in beta_grid:
        if beta+alpha<.99:
            val = dcc_nll_fixed((alpha, beta),
                                      ccc.garch_volatilities,
                                      ccc.returns
                                      )
            if val<2e5:
                tempo.append(val)
            else:
                tempo.append(np.nan)
        else:
            tempo.append(np.nan)
    values.append(tempo)
values = pd.DataFrame(values, columns = beta_grid, index = alpha_grid)

sns.heatmap(values, vmax=None)
plt.show()

fig = go.Figure(data=[go.Surface(z=values.values, x = beta_grid, y = alpha_grid)])
fig.update_layout(
    scene = dict(
        xaxis = dict(
            title='beta'),
        yaxis = dict(
            title='alpha'),
        zaxis = dict(
            title='DCC Loss Function')
        )
    )
plot(fig)
