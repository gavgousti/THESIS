# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:42:55 2022

@author: gavgous
"""

import yfinance as yf, pandas as pd, tensorflow as tf, numpy as np
from arch import arch_model
from UTILS import train_test_split_ts, forward_garch, take_X_y, forward_gjr, forward_egarch
import matplotlib.pyplot as plt

data = yf.Ticker('^GSPC').history(start = '2000-01-01').Close
data = 100*data.pct_change().dropna()
train, test = train_test_split_ts(data)

X_train, y_train = take_X_y(train, 20, reshape = True, take_rv = True, log_rv =True )
X_test, y_test = take_X_y(test, 20, reshape = True, take_rv = True, log_rv =True)

X_train, y_train = tf.convert_to_tensor(X_train, dtype  = 'float32'),\
    tf.convert_to_tensor(y_train, dtype  = 'float32') 

X_test, y_test = tf.convert_to_tensor(X_test, dtype  = 'float32'),\
    tf.convert_to_tensor(y_test, dtype  = 'float32') 


# =============================================================================
# GARCH MODEL
# =============================================================================

garch = arch_model(y_train, mean = 'Constant', vol = 'GARCH', p=1, q=1)
fit = garch.fit(disp = False)


output = forward_garch(y_test, y_train, fit)
plt.plot(output)


# =============================================================================
# GJR Model
# =============================================================================

gjr = arch_model(y_train, p=1, o=1, q=1)
fit_gjr = gjr.fit(disp=False)

output2 = forward_gjr(y_test, y_train, fit_gjr)
plt.plot(output2)
output2

# =============================================================================
# EGARCH
# =============================================================================

egarch = arch_model(y_train, p = 1, q = 1, o = 1, vol = 'EGARCH')
fit_egarch = egarch.fit(disp=False)

output3 = forward_egarch(y_test, y_train, fit_egarch)
plt.plot(output3)

