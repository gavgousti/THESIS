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
from UTILS import take_pairs, nll_md

cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)


data = pd.read_csv(r"C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Data\SMI_data.csv", index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

ccc = CCC(data, scale = 100)
ccc.fit()

sns.heatmap(ccc.P_c, cmap = 'viridis')
plt.title('Constant Correlation')
plt.show()

ccc.plot_cov_feature('det')
ccc.plot_cov_feature('trace')
ccc.plot_cov_feature('sum')

print(ccc.c_nll)

def dcc_c_nll(
        alpha,
        beta, 
        sigma,
        P_c,
        x
        ):
    '''
    Calculated a negative log likelihood of p-dimensional returns.
    Used for the DCC model.

    Parameters
    ----------
    alpha : float
        alpha parameter of DCC.
    beta : float
        beta parameter of DCC..
    sigma : pd.DataFrame
        Conditional volatilities of the Garch models. Shape = (T, p)
    P_c : pd.DataFrame
        Constant Conditional Correlation of shape (p,p).
    x : pd.DataFrame
        Returns of shape (T,p).

    Returns
    -------
    float
        Negative log-likelihood value.

    '''
    P_c = P_c.values
    prev_corr = P_c.copy()
    val = 0
    for t in range(x.shape[0]):
        x_ = x.iloc[0].values.reshape(-1,1)
        sigma_ = sigma.iloc[t].values    
        
        Sigma_t = np.diag(sigma_)@\
            ((1-alpha-beta)*P_c+alpha*x_@np.transpose(x_)+beta*prev_corr)@\
                np.diag(sigma_)
                
        val+=np.log(np.linalg.det(Sigma_t)) +\
            np.transpose(x_)@np.linalg.inv(Sigma_t)@x_
            
        prev_corr = np.diag(1/sigma_)@Sigma_t@np.diag(1/sigma_)
    return .5*val[0][0]

alpha = .3; beta = .3
sigma = ccc.garch_volatilities.iloc[:, [0,1]]
P_c = ccc.P_c.iloc[[0,1], [0,1]]
x = ccc.returns.iloc[:, [0,1]]

dcc_c_nll(alpha, beta, sigma, P_c, x)

def dcc_comp_nll(
        alpha,
        beta,
        volatilities,
        stock_names,
        X,
        P_c
        ):
    #TODO: docstring
    '''
    

    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    volatilities : TYPE
        DESCRIPTION.
    stock_names : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    P_c : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    val = 0 
    pairs_cont = take_pairs(
        np.arange(stock_names.shape[0]),
        'contiguous'
        )
    for pair in tqdm(pairs_cont):
        val+=dcc_c_nll(
            alpha,
            beta,
            volatilities.iloc[:, pair],
            P_c.iloc[pair,:].iloc[:, pair],
            X.iloc[:, pair])
    return val/len(pairs_cont)
    

dcc_comp_nll(.01,
             .01,
             ccc.garch_volatilities,
             ccc.stock_names,
             ccc.returns,
             ccc.P_c
             )
ccc.c_nll

#TODO: why different value when alpha, beta = 0,0?
#TODO: optimization