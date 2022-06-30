# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:40:00 2022

@author: Giorgio
"""
# TODO: docstrings

import tensorflow as tf, tensorflow.math as m, numpy as np, pandas as pd
import math 
from tensorflow.keras.utils import timeseries_dataset_from_array as loader
from arch import arch_model
from tqdm import tqdm
pi = tf.constant(math.pi)

# =============================================================================
# add:df.set_index(_pd.DatetimeIndex(df.index), inplace = True):line247,history
# =============================================================================

def create_dataset_pct(paths, closeout):
    output = []
    for t in range(0, paths.shape[1]-closeout):
        output.append((paths.iloc[:,t+closeout]-paths.iloc[:,t])/paths.iloc[:,t])
    output = 100*pd.DataFrame(output)
    return output

def create_dataset(paths, closeout = 10):
    '''
    Function that takes the simulated paths of the DCC and transforms it to 
    the log returns with a specified lag.

    Parameters
    ----------
    paths : pd.DataFrame
        Data frame of shape (simulations, horizon). The portfolio paths
    closeout : int, optional
        The closeout period for the model. The default is 10.

    Returns
    -------
    pd.DataFrame
        The transformed log returns.

    '''
    output = []
    paths = np.log(paths)
    for t in range(0, paths.shape[1]-closeout):
        output.append((paths.iloc[:,t+closeout]-paths.iloc[:,t]))
    return pd.DataFrame(output)


def simulate_garch(
        params,
        simulations = 100,
        horizon = 10
        ):
    '''
    Function that simulates GARCH paths.

    Parameters
    ----------
    params : tuple
        Garch parameters: omega, alpha, beta.
    simulations : int, optional
        Number of simulated paths. The default is 100.
    horizon : int, optional
        Time horizon for every path. The default is 10.

    Returns
    -------
    dict
        A dictionary with keys "Volatilities" and "Residuals" with the
        simulated volatilities and residuals respectively.

    '''
    omega, alpha, beta = params.omega, params['alpha[1]'], params['beta[1]']
    output_vola = []
    output_res = []
    for _ in range(simulations):
        # print(omega/(1-alpha-beta+1e-3))
        vola_path = [np.sqrt(omega/(1-alpha-beta))]
        r = np.random.randn()*vola_path[-1]
        res = [r/vola_path[-1]]
        for _ in range(horizon):
            vola_path.append(np.sqrt(omega\
                                     +alpha*(r/vola_path[-1])**2\
                                         +beta*vola_path[-1]**2))
            r = np.random.randn()*vola_path[-1]
            res.append(r/vola_path[-1])
        output_vola.append(vola_path)
        output_res.append(res)
    output_vola =  pd.DataFrame(output_vola, columns=['h'+str(i) for i in range(horizon+1)])
    output_res =  pd.DataFrame(output_res, columns=['h'+str(i) for i in range(horizon+1)])
    return {'Volatilities': output_vola,
            'Residuals': output_res}

def cov_to_corr(Q_t):
        P_t = np.diag(1/np.sqrt(np.diag(Q_t)))@\
            Q_t@\
                np.diag(1/np.sqrt(np.diag(Q_t)))
        return P_t
    
def forward_CC(
        model,
        test
        ):
    '''
    Forward pass of a CC model

    Parameters
    ----------
    model : thesis.CC
        CC model.
    test : pd.DataFrame
        PRCES of the test set.

    Returns
    -------
    tuple
        (covariances, nll).

    '''
    returns_test = test.pct_change().dropna()
    volatilities_forward = forward_garch_multi(
        test,
        model.data,
        model.garch_models,
        model.vola_type
        )
    covariances_forward = np.full(
        (returns_test.shape[0],
         returns_test.shape[1],
         returns_test.shape[1]),
        .0
        )
    alpha, beta = model.theta
    Q_bar = np.divide(returns_test.iloc[41:],
                      volatilities_forward).cov().values
    Q_t = Q_bar.copy()
    for t in tqdm(range(41, returns_test.shape[0])):
        x_ = returns_test.iloc[t].values.reshape(-1,1)
        sigma_ = volatilities_forward.iloc[t-41].values.reshape(-1,1)
        eps = np.divide(x_, sigma_)
        
        Q_t = (1-alpha-beta)*Q_bar+\
            alpha*eps@eps.transpose()+\
                beta*Q_t
        
        P_t = np.diag(1/np.sqrt(np.diag(Q_t)))@\
            Q_t@\
                np.diag(1/np.sqrt(np.diag(Q_t)))
                
        Sigma_t = np.diag(sigma_.ravel())@P_t@np.diag(sigma_.ravel())
                
        covariances_forward[t, :, :] = Sigma_t
    return covariances_forward[41:], dcc_nll((alpha, beta),
                                        volatilities_forward,
                                        returns_test.iloc[41:])

def forward_garch_multi(
        test,
        train,
        fits,
        vola_type
        ):
    '''
    Forward pass of 1d garch models for the multidimensional setup.

    Parameters
    ----------
    test : pd.DataFrame
        test set of PRICES of shape (m, d).
    train : pd.DataFrame
        train set of PRICES of shape (n, d).
    fits : dict
        dictionary with the collected garch models,
        e.g. model.garch_models.
    vola_type: str
        type of volatility model {'GARCH', 'EGARCH', 'GJR'}

    Returns
    -------
    pd.DataFrame
        Forwad Conditional Volatilities of shape (m-1, d).

    '''
    test_volatilities = []
    if vola_type=='GARCH':
        forward_1d = forward_garch
    elif vola_type=='EGARCH':
        forward_1d = forward_egarch 
    elif vola_type=='GJR':
        forward_1d = forward_gjr
    #TODO: GB, etc
        
    for stock in tqdm(range(len(fits))):
        
        fit = list(fits.values())[stock]

        if vola_type == 'GB':
            rets_test = 100*test.iloc[:, stock].pct_change().dropna()
            test_volatilities.append(fit.predict(rets_test).values.\
                                     tolist())
        
        elif vola_type == 'FNN' or vola_type == 'LSTM':
            rets_test = 100*test.iloc[:, stock].pct_change().dropna()
            lag =20
            X_test, y_test = take_X_y(rets_test,
                                        lag,
                                        reshape = vola_type=='FNN',
                                        take_rv = True,
                                        log_rv =True)

            X_test, y_test = tf.convert_to_tensor(
                X_test,
                dtype  = 'float32'
                ),\
                tf.convert_to_tensor(
                    y_test,
                    dtype  = 'float32'
                    )
            test_volatilities.append(fit(X_test).numpy().ravel().tolist())
        
        elif vola_type in ['GARCH', 'EGARCH', 'GJR']:            
            rets_train = tf.convert_to_tensor(100*train.iloc[:,stock].pct_change().\
                                              dropna().values.reshape(-1,1),\
                                                  dtype  = 'float32')
            rets_test = tf.convert_to_tensor(100*test.iloc[:,stock].pct_change().\
                                             dropna().values.reshape(-1,1),\
                                                 dtype  = 'float32')
                
            test_volatilities.append(forward_1d(rets_test, rets_train, fit).\
                           numpy().ravel().tolist()[41:])
                
    return pd.DataFrame(data = test_volatilities,
                        columns = test.index[42:],
                        index = test.columns).transpose()

def take_DCC_cov(
        theta,
        sigma,
        P_c,
        x):
    '''
    Helper function for obtaining the covariances in the DCC model.

    Parameters
    ----------
    theta : tuple
        (alpha, beta).
    sigma : pd.DataFrame
        Individual volatilities.
    P_c : pd.DataFrame
        Constant Correlation matrix.
    x : pd.DataFrame
        Assets' returns.

    Returns
    -------
    cov : np.array
        3d array of shape (T, d, d) storing the covariances.

    '''
    cov = np.full((x.shape[0], x.shape[1], x.shape[1]), .0)
    alpha, beta = theta
    Q_bar = np.divide(x, sigma).cov().values
    Q_t = Q_bar.copy()
    for t in tqdm(range(1, x.shape[0])):
        x_ = x.iloc[t-1].values.reshape(-1,1)
        sigma_ = sigma.iloc[t-1].values.reshape(-1,1)
        eps = np.divide(x_, sigma_)
        
        Q_t = (1-alpha-beta)*Q_bar+\
            alpha*eps@eps.transpose()+\
                beta*Q_t
        
        P_t = np.diag(1/np.sqrt(np.diag(Q_t)))@\
            Q_t@\
                np.diag(1/np.sqrt(np.diag(Q_t)))
                
        Sigma_t = np.diag(sigma_.ravel())@P_t@np.diag(sigma_.ravel())
                
        cov[t, :, :] = Sigma_t
    return cov, Q_t

def dcc_nll(
        theta,
        sigma,
        x
        ):
    alpha, beta = theta
    Q_bar = np.divide(x, sigma).cov().values
    Q_t = Q_bar.copy()
    nll = 0
    for t in range(1, x.shape[0]):
        x_ = x.iloc[t-1].values.reshape(-1,1)
        sigma_ = sigma.iloc[t-1].values.reshape(-1,1)
        eps = np.divide(x_, sigma_)
        
        Q_t = (1-alpha-beta)*Q_bar+\
            alpha*eps@eps.transpose()+\
                beta*Q_t
                        
                
        P_t = np.diag(1/np.sqrt(np.diag(Q_t)))@\
            Q_t@\
                np.diag(1/np.sqrt(np.diag(Q_t)))
                
        Sigma_t = np.diag(sigma_.ravel())@P_t@np.diag(sigma_.ravel())
        
        nll+= .5*(np.log(np.linalg.det(Sigma_t)) +\
            np.transpose(x_)@np.linalg.inv(Sigma_t)@x_)
    return nll[0,0]
    
    
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
    """
    Conditional Volatility of a Garch(1,1) model on the test set.

    Parameters
    ----------
    rets_test : tf.Tensor
        test set of shape (n,1).
    rets_train : tf.Tensor
        train set of shape (m,1).
    fit : arch.univariate.base.ARCHModelResult
        the fitted model on the train set.

    Returns
    -------
    tf.Tensor
        conditional volatility on the test set of shape (n,1).

    """
    mu, omega, alpha, beta = fit.params
    print(mu, omega, alpha, beta)
    eps = rets_test-mu
    eps_0 = (rets_train[-1]-mu).numpy(); eps_0 = tf.convert_to_tensor(eps_0.reshape(-1,1), dtype = 'float32')
    eps = tf.concat((eps_0, eps[:-1,:]),0)
    sigma2 = [fit.conditional_volatility[-1]**2]
    for t in range(rets_test.shape[0]):
        sigma2.append((omega+alpha*eps[t]**2+beta*sigma2[-1]).numpy()[0])
    return tf.expand_dims(tf.convert_to_tensor(sigma2[1:], dtype = 'float32')**.5,1)

def forward_gjr(rets_test, rets_train, fit):
    """
    Conditional Volatility of a GJR(1,1,1) model on the test set.

    Parameters
    ----------
    rets_test : tf.Tensor
        test set of shape (n,1).
    rets_train : tf.Tensor
        train set of shape (m,1).
    fit : arch.univariate.base.ARCHModelResult
        the fitted model on the train set.

    Returns
    -------
    tf.Tensor
        conditional volatility on the test set of shape (n,1).

    """
    mu, omega, alpha, gamma, beta = fit.params
    eps = rets_test-mu
    eps_0 = (rets_train[-1]-mu).numpy(); eps_0 = tf.convert_to_tensor(eps_0.reshape(-1,1), dtype = 'float32')
    eps = tf.concat((eps_0, eps[:-1,:]),0)
    sigma2 = [fit.conditional_volatility[-1]**2]
    for t in range(rets_test.shape[0]):
        gjr_term = gamma*eps[t]**2*(eps[t].numpy()[0]<0)
        sigma2.append((omega+alpha*eps[t]**2+beta*sigma2[-1]).numpy()[0]+gjr_term.numpy()[0])
    return tf.expand_dims(tf.convert_to_tensor(sigma2[1:], dtype = 'float32')**.5,1)

def forward_egarch(rets_test, rets_train, fit):
    """
    Conditional Volatility of an assymetric EGARCH(1,1,1) model on the test set.

    Parameters
    ----------
    rets_test : tf.Tensor
        test set of shape (n,1).
    rets_train : tf.Tensor
        train set of shape (m,1).
    fit : arch.univariate.base.ARCHModelResult
        the fitted model on the train set.

    Returns
    -------
    tf.Tensor
        conditional volatility on the test set of shape (n,1).

    """
    mu, omega, alpha, gamma, beta = fit.params
    eps = rets_test-mu
    eps_0 = (rets_train[-1]-mu).numpy(); eps_0 = tf.convert_to_tensor(eps_0.reshape(-1,1), dtype = 'float32')
    eps = tf.concat((eps_0, eps[:-1,:]),0)
    log_sigma2 = [np.log(fit.conditional_volatility[-1]**2)]
    for t in range(rets_test.shape[0]):
        e_t = eps[t].numpy()[0]/np.exp(log_sigma2[-1])**.5
        log_sigma2.append((omega\
                           + alpha*(np.abs(e_t)-np.sqrt(1/np.pi)) \
                           + beta*log_sigma2[-1])\
                           + gamma*e_t)
    return tf.expand_dims(np.exp(tf.convert_to_tensor(log_sigma2[1:], dtype = 'float32'))**.5,1)

def nll_gb_exp(y_hat, data):
    #y_hat = log(sigma**2)
    y = data.get_label()
    grad = .5*(np.ones_like(y) - y**2*np.exp(-1*y_hat))
    hess = .5*(y**2*np.exp(-1*y_hat))
    return grad, hess

def nll_gb_exp_eval(y_hat, data):
    y = data.get_label()
    return 'NLL', .5*np.sum(np.log(2*np.pi)+y_hat+y**2*np.exp(-1*y_hat)), False


def take_X_y(rets, lag, take_rv, reshape, **kwargs):
    if not take_rv:
        rets = rets.values-rets.mean()
        load = loader(
            data = rets, 
            targets = rets[lag:],
            sequence_length = lag,
            batch_size = rets.shape[0]
        )
        for X, y in load:
            return  X.numpy().astype('float32'), y.numpy().reshape(-1,1).astype('float32')
    else:
        rets = rets-rets.mean()
        rv = rets.rolling(22).std().dropna().values.reshape(-1,1)
        if kwargs['log_rv']:
            rv = np.log(rv)
            print('log-realized volatility used as additional feature!')
        rets = rets.iloc[21:].values.reshape(-1,1)
        data_aug = np.concatenate((rets, rv), 1)
        load = loader(
            data = data_aug,
            targets = rets[lag:],
            sequence_length = lag,
            batch_size = data_aug.shape[0]
        )
        for X, y in load:
            X, y = X.numpy().astype('float32'), y.numpy().astype('float32')
        if reshape:
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
        return X, y
