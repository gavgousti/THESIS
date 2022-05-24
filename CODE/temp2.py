# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:32:30 2022

@author: gavgous
"""

# =============================================================================
# TEMPORARAY SCRIPT
# =============================================================================


from thesis import CC
from sklearn.metrics import mean_squared_error as mse
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl, tensorflow as tf
from tqdm import tqdm
from UTILS import dcc_nll, take_DCC_cov, train_test_split_ts, forward_garch_multi
from scipy.optimize import minimize
import plotly.graph_objects as go, seaborn as sns
from plotly.offline import plot
from arch import arch_model
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)

path = r"C:\Users\gavgous\OneDrive - Zanders-BV\Desktop\THESIS\GITHUB\DATA"

# =============================================================================
# available data sets:
#    - FTSE_dat.csv
#    - SMI_data.csv
#    - Berkshire_Hathaway_Portfolio_data_.csv
# =============================================================================

data = pd.read_csv(path+"\\SMI_data.csv", index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

train, test = train_test_split_ts(data)

dcc = CC(train, correlation_type = 'Dynamic')
dcc.fit()
resultsDCC = dcc.evaluation_on_test_set(test)

ccc = CC(train, correlation_type='Constant')
ccc.fit()
resultsCCC = ccc.evaluation_on_test_set(test)


#TODO: ML volatilities


# =============================================================================
# what if we estimate Q_t from ml?
# example:
#     n,p = train.shape # = (2174, 19)
#     output of estimator = int(n*(.5*p**2+.5*p)) # = 190
# =============================================================================
# =============================================================================
# NN example:
#         
#     compl =  lambda x: 2*(x+1)+(x+1)*190
#     x = np.arange(50)
#     plt.plot(x, compl(x))
# =============================================================================


cov_vals = pd.DataFrame(pd.unique(dcc.covariances.ravel()))
cov_vals.describe(percentiles = [0, .05, .25, .5, .75, .95, .1])

plt.hist(cov_vals, bins = 3000)
plt.xlim(-4, 10)

