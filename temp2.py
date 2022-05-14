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
import time, numpy as np
from tqdm import tqdm
from UTILS import take_all_pairs



data = pd.read_csv('SMI_data.csv', index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

ccc = CCC(data, scale = 30)
ccc.fit()

sns.heatmap(ccc.P_c, cmap = 'viridis')
plt.title('Constant Correlation')
plt.show()

ccc.plot_cov_feature('det')
ccc.plot_cov_feature('trace')
ccc.plot_cov_feature('sum')

ccc.check_pd()


def nll_2d(covariances, returns):
    s = 0 
    for t in (range(returns.shape[0])):
        x = returns.iloc[t].values.reshape(-1,1)
        s+=.5*(np.log(np.linalg.det(covariances[t]))+np.transpose(x)@\
               np.linalg.inv(covariances[t])@x)
    return s
    
def take_2d_covariance(covariances, pair):
    cov = {}
    for t in range(len(covariances)):
        cov[t] = covariances[t].loc[pair, pair]
    return cov

pairs = take_all_pairs(ccc.stock_names)

cov2 = take_2d_covariance(ccc.conditional_covariances, pairs[0])


# l = 0
# for pair in tqdm(pairs):
#     start = time.time()
#     cov = take_2d_covariance(ccc.conditional_covariances, pair)
#     print('Cov calc in {}\n'.format(time.time()-start))
#     start = time.time()
#     l+=nll_2d(cov, ccc.returns.loc[:,pair])
#     print('NLL calc in {}'.format(time.time()-start))
# l/=len(pairs)


# =============================================================================
# IMPORTANT PART TO BE IMPLEMENTED INSIDE CCC
# =============================================================================
covs = []
for t in tqdm(range(ccc.returns.shape[0])):
    covs.append(ccc.conditional_covariances[t].values)
    
    
covs = np.array(covs)

covs[:, 1:3, 1:3].shape
