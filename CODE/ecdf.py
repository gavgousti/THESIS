# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:12:51 2022

@author: gavgous
"""

# =============================================================================
# monte carlo 1d test
# =============================================================================

import yfinance as yf
from arch import arch_model
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
rets = 100*yf.Ticker('^GSPC').history(start='1990-01-01').Close.pct_change().dropna()

garch = arch_model(rets, mean = 'zero', vol = 'GARCH')
garch.fit()
omega, alpha, beta = garch.fit().params

simulations = 10000; T = 600
out = []
for _ in tqdm(range(simulations)):
    sigma = rets.std()
    path = [sigma*np.random.randn()]
    for _ in range(T):
        sigma = np.sqrt(omega+alpha*path[-1]**2/sigma**2+beta*sigma**2)
        path.append(sigma*np.random.randn())
    out.append(path)
out = pd.DataFrame(out)

total = out.values.ravel()

ecdf = ECDF(total)
x = np.linspace(-3, 3, 100)
plt.plot(x, ecdf(x))
plt.show()


slope_changes = sorted(set(total))

sample_edf_values_at_slope_changes = [ ecdf(item) for item in slope_changes]
inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)


plt.plot(np.linspace(.001, .999, 1000), inverted_edf(np.linspace(.001, .999, 1000)))
plt.show()
