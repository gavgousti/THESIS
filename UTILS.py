# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:40:00 2022

@author: Giorgio
"""

import tensorflow as tf, tensorflow.math as m, numpy as np, pandas as pd
import math 
from arch import arch_model
pi = tf.constant(math.pi)

def nll(sigma2, r):
    return .5*m.reduce_sum(m.log(2*pi) + m.log(sigma2) + m.divide(r**2, sigma2))

def train_test_split_ts(ts, pct=.7):
    return (ts.iloc[:int(pct*ts.shape[0])], ts.iloc[int(pct*ts.shape[0]):])

def prepare_data_(rets, p=1, q=1, log_vola = True):
    rets -= rets.mean()
    rets = rets.values
    garch = arch_model(rets, mean = 'Constant', vol = 'GARCH', p=1, q=1)
    fit = garch.fit(disp = False)
    garch_vola = fit.conditional_volatility
    if log_vola:
        garch_vola = np.log(garch_vola)
    rets_cols = []
    for i in range(p):
        rets_cols.append(rets[i:rets.shape[0]-p+i])
    prev_rets = np.array(rets_cols).transpose()
    vola_cols = []
    for j in range(q):
        vola_cols.append(garch_vola[j:garch_vola.shape[0]-q+j])
    prev_vola = np.array(vola_cols).transpose()
    data = np.concatenate((rets[p:rets.shape[0]].reshape(-1,1), prev_rets, prev_vola), 1)
    data = tf.convert_to_tensor(data, dtype = 'float32')
    
    return tf.reshape(data[:,0], (data[:,0].shape[0],1)), data[:,1:]

def rolling(days, x):
    return tf.convert_to_tensor(pd.DataFrame(x[:,0]).rolling(days).mean().dropna().values.reshape(-1,1),
                     dtype = 'float32')

def forward_garch(rets_test, rets_train, fit):
    mu, omega, alpha, beta = fit.params
    eps = rets_test-mu
    eps_0 = (rets_train[-1]-mu).numpy(); eps_0 = tf.convert_to_tensor(eps_0.reshape(-1,1), dtype = 'float32')
    eps = tf.concat((eps_0, eps[:-1,:]),0)
    sigma2 = [fit.conditional_volatility[-1]**2]
    for t in range(rets_test.shape[0]):
        sigma2.append((omega+alpha*eps[t]**2+beta*sigma2[-1]).numpy()[0])
    return tf.expand_dims(tf.convert_to_tensor(sigma2[1:], dtype = 'float32')**.5,1)