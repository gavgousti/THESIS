# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:51 2022

@author: Giorgio
"""

from tensorflow import keras
from keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, LSTM
from tqdm import tqdm
import matplotlib as mpl
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from UTILS import train_test_split_ts, forward_garch, nll_gb_exp, nll_gb_exp_eval, take_X_y,nll

import pyfiglet
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
from arch import arch_model

mpl.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')

class DNN(keras.Model):
    
    def __init__(
        self,
        hidden_size = [100,],
        hidden_activation = 'tanh',
        last_activation = 'exponential',
        dropout = 0,
        l1 = 0,
        l2 = 0
    ):
        super().__init__()
        self.__layers = []
        for l in hidden_size:
            self.__layers.append(BatchNormalization())
            self.__layers.append(Dense(l, activation = hidden_activation))
            self.__layers.append(Dropout(dropout))
        self.__layers.append(Dense(1, activation = last_activation))
        
    def call(
        self,
        X
    ):
        for f in self.__layers:
            X = f(X)
        return X
    
    def train(
        self,
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs,
        bs,
        lr
    ):
        self.loss_train = []; self.loss_val = []
        self.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = nll)
        for epoch in tqdm(range(1, epochs+1)):
            for i in range(X_train.shape[0]//bs):
                X_, y_ = X_train[i*bs: i*bs+bs], y_train[i*bs: i*bs+bs]
                if i ==X_train.shape[0]//bs-1:
                    X_, y_ = X_train[i*bs+bs:X_train.shape[0]], y_train[i*bs+bs:X_train.shape[0]]
                with tf.GradientTape() as tape:
                    logits = self(X_)
                    loss = nll((logits)**2, y_)
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_train.append(nll(self(X_train)**2, y_train))
            self.loss_val.append(nll(self(X_test)**2, y_test))
            if epoch%5==0:
                print('EPOCH: {}\n'.format(epoch)+30*'-'+'\nTRAIN LOSS: {:5.0f}\nTEST LOSS: {:5.0f}'\
                      .format(self.loss_train[-1], self.loss_val[-1]))
                print(30*'=')
    
    def plot_loss(
        self
    ):
        plt.plot(self.loss_train, label = 'TRAIN')
        plt.plot(self.loss_val, label = 'TEST')
        plt.legend()
        plt.title('NLL')
        plt.show()      

class RNN(keras.Model):
    
    def __init__(
        self,
        lstm = False,
        hidden_size = [100,],
        hidden_activation = 'tanh',
        last_activation = 'exponential',
        dropout = 0,
        l1 = 0,
        l2 = 0
    ):
        super().__init__()
        self.__layers = []
        self.__layers.append(BatchNormalization())
        if lstm:
            self.NAME = 'LSTM Recurrent Neural Network'
            self.__layers.append(LSTM(hidden_size[0], activation = hidden_activation))
        else:
            self.NAME = 'Simple Recurrent Neural Network'
            self.__layers.append(SimpleRNN(hidden_size[0], activation = hidden_activation))            
        for l in hidden_size[1:]:
            self.__layers.append(BatchNormalization())
            self.__layers.append(Dense(l, activation = hidden_activation))
            self.__layers.append(Dropout(dropout))
        self.__layers.append(Dense(1, activation = last_activation))
        
    def call(
        self,
        X
    ):
        for f in self.__layers:
            X = f(X)
        return X
    
    def train(
        self,
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs,
        bs,
        lr
    ):
        self.loss_train = []; self.loss_val = []
        self.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = nll)
        for epoch in tqdm(range(1, epochs+1)):
            for i in range(X_train.shape[0]//bs):
                X_, y_ = X_train[i*bs: i*bs+bs], y_train[i*bs: i*bs+bs]
                if i ==X_train.shape[0]//bs-1:
                    X_, y_ = X_train[i*bs+bs:X_train.shape[0]], y_train[i*bs+bs:X_train.shape[0]]
                with tf.GradientTape() as tape:
                    logits = self(X_)
                    loss = nll((logits)**2, y_)
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_train.append(nll(self(X_train)**2, y_train))
            self.loss_val.append(nll(self(X_test)**2, y_test))
            if epoch%5==0:
                print('EPOCH: {}\n'.format(epoch)+30*'-'+'\nTRAIN LOSS: {:5.0f}\nTEST LOSS: {:5.0f}'\
                      .format(self.loss_train[-1], self.loss_val[-1]))
                print(30*'=')
    
    def plot_loss(
        self
    ):
        plt.plot(self.loss_train, label = 'TRAIN')
        plt.plot(self.loss_val, label = 'TEST')
        plt.legend()
        plt.title('NLL')
        plt.show() 

def deployment_RNN_1d(
        lstm = False,
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None
        ):
    """
    # TBA

    Parameters
    ----------
    lstm : TYPE, optional
        DESCRIPTION. The default is False.
    index : TYPE, optional
        DESCRIPTION. The default is '^GSPC'.
    start_date : TYPE, optional
        DESCRIPTION. The default is '2000-01-01'.
    lag : TYPE, optional
        DESCRIPTION. The default is 20.
    include_rv : TYPE, optional
        DESCRIPTION. The default is True.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    prices = yf.Ticker(index).history(start = start_date).Close
    rets = 100*(prices.pct_change().dropna())
    rets_train, rets_test = train_test_split_ts(rets, .7)
    X_train, y_train = take_X_y(rets_train, lag, reshape = False, take_rv = include_rv, log_rv =include_rv )
    X_test, y_test = take_X_y(rets_test, lag, reshape = False, take_rv = include_rv, log_rv =include_rv)
    
    X_train, y_train = tf.convert_to_tensor(X_train, dtype  = 'float32'),\
                            tf.convert_to_tensor(y_train, dtype  = 'float32') 
    
    X_test, y_test = tf.convert_to_tensor(X_test, dtype  = 'float32'),\
                            tf.convert_to_tensor(y_test, dtype  = 'float32') 
    
    model = RNN(
        lstm = lstm,
        hidden_size = [60],
        hidden_activation = 'tanh',
        last_activation = 'exponential',
        dropout = 0.0,
        l1 = 5,
        l2 = 0
    )
    
    model.train(
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs = 10,
        bs = 2048,
        lr = .008
    )
    
    model.plot_loss()
    
    print(model.summary())
    
    print(60*'*')
    print(pyfiglet.figlet_format("             MODEL\nEVALUATION"))
    print(60*'*')
    
    out = model(X_train)
    plt.plot(out)
    plt.title(model.NAME)
    plt.show()
    
    print('\n\nPerformance on the Train Set:\n'+30*'-'+'\n')
    print('RNN NLL: {:6.0f}'.format(nll(out**2, y_train)))
    
    garch = arch_model(y_train, mean = 'Constant', vol = 'GARCH', p=1, q=1)
    fit = garch.fit(disp = False)
    g_vol = fit.conditional_volatility
    
    plt.plot(g_vol)
    plt.title('GARCH')
    plt.show()
    print('Garch NLL: {:6.0f}'.format(nll(tf.convert_to_tensor(g_vol.reshape(-1,1), dtype = 'float32')**2,
                                          y_train)))
    
    print('\n\nPerformance on the Test Set:\n'+30*'-'+'\n')

    out_test = model(X_test) 
    plt.plot(np.exp(X_test[:,-1,1]), label = 'Realized Volatility', alpha = 1)
    plt.title(model.NAME+' Out of Sample '+ index)
    
    if lstm:
        txt = 'LSTM'

    else:
        txt = 'RNN'
        
    if save!=None:
        plt.plot(out_test, label = txt)
        plt.legend()
        plt.savefig(save+'\\'+txt+'__'+index+'.png')
        plt.show()
    else:
        plt.plot(out_test, label = txt)
        plt.legend()
        plt.show()
        

        
    
    print('RNN NLL: {:6.0f}'.format(nll(out_test**2, y_test)))
    print('RNN RMSE: {:1.3f}'.format(mse(np.exp(X_test[:,-1,1]), out_test.numpy().ravel())**.5))
    
    
    g_vola_pred = forward_garch(y_test, y_train, fit)
    plt.plot(g_vola_pred, label = 'GARCH')
    plt.plot(np.exp(X_test[:,-1,1]), label = 'Realized Volatility', alpha = 1)
    plt.title('GARCH Out of Sample '+ index)
    plt.legend()
    plt.show()
    # if save!=None:
    #     plt.savefig(save+'\\GARCH__'+index+'.png')
        
    print('Garch NLL: {:6.0f}'.format(nll(g_vola_pred**2, y_test)))
    print('Garch RMSE: {:1.3f}'.format(mse(np.exp(X_test[:,-1,1]), g_vola_pred.numpy().ravel())**.5))
    return {'name': txt+'__'+index,
            'NLL': nll(out_test**2, y_test), 
            'RMSE': mse(np.exp(X_test[:,-1,1]), out_test.numpy().ravel())**.5}


def deployment_DNN_1d(
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None
        ):
    """
    

    Parameters
    ----------
    index : TYPE, optional
        DESCRIPTION. The default is '^GSPC'.
    start_date : TYPE, optional
        DESCRIPTION. The default is '2000-01-01'.
    lag : TYPE, optional
        DESCRIPTION. The default is 20.
    include_rv : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    
    prices = yf.Ticker(index).history(start = start_date).Close
    rets = 100*(prices.pct_change().dropna())
    rets_train, rets_test = train_test_split_ts(rets, .7)
    X_train, y_train = take_X_y(rets_train, lag, reshape = True, take_rv = include_rv, log_rv =include_rv )
    X_test, y_test = take_X_y(rets_test, lag, reshape = True, take_rv = include_rv, log_rv =include_rv)
    
    X_train, y_train = tf.convert_to_tensor(X_train, dtype  = 'float32'),\
                            tf.convert_to_tensor(y_train, dtype  = 'float32') 
    
    X_test, y_test = tf.convert_to_tensor(X_test, dtype  = 'float32'),\
                            tf.convert_to_tensor(y_test, dtype  = 'float32') 
    
    model = DNN(
        hidden_size = [300],
        dropout = .5,
        l1 = 1,
        l2 = 1
    )
    
    model.train(
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs = 20,
        bs = 1024,
        lr = .001,
    )
    
    model.plot_loss()
    
    model.summary()
    
    print(60*'*')
    print(pyfiglet.figlet_format("             MODEL\nEVALUATION"))
    print(60*'*')
    
    out = model(X_train)
    plt.plot(out)
    plt.title('Feed-forward Neural Network')
    plt.show()
    
    print('\n\nPerformance on the Train Set:\n'+30*'-'+'\n')
    print('DNN NLL: {:6.0f}'.format(nll(out**2, y_train)))
    
    garch = arch_model(y_train, mean = 'Constant', vol = 'GARCH', p=1, q=1)
    fit = garch.fit(disp = False)
    g_vol = fit.conditional_volatility
    
    plt.plot(g_vol)
    plt.title('GARCH')
    plt.show()
    print('Garch NLL: {:6.0f}'.format(nll(tf.convert_to_tensor(g_vol.reshape(-1,1), dtype = 'float32')**2,
                                          y_train)))
    
    print('\n\nPerformance on the Test Set:\n'+30*'-'+'\n')

    out_test = model(X_test) 
    plt.plot(out_test, label = 'FNN')
    plt.plot(np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatility')
    plt.title('Feed-forward Neural Network Out of Sample '+ index)
    plt.legend()
    if save!=None:
        plt.savefig(save+'\\FNN__'+index+'.png')
        plt.show()

    
    print('DNN NLL: {:6.0f}'.format(nll(out_test**2, y_test)))
    print('DNN RMSE: {:1.3f}'.format(mse(np.exp(X_test[:,-1]), out_test.numpy().ravel())**.5))
    
    
    g_vola_pred = forward_garch(y_test, y_train, fit)
    plt.plot(g_vola_pred, label = 'GARCH')
    plt.plot(np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatility')
    plt.title('GARCH Out of Sample '+ index)
    plt.legend()
    plt.show()
    # if save!=None:
    #     plt.savefig(save+'\\GARCH__'+index+'.png')

    print('Garch NLL: {:6.0f}'.format(nll(g_vola_pred**2, y_test)))
    print('GARCH RMSE: {:1.3f}'.format(mse(np.exp(X_test[:,-1]),g_vola_pred)**.5))
    
    return {'name': 'FNN__'+index,
            'NLL': nll(out_test**2, y_test),
            'RMSE': mse(np.exp(X_test[:,-1]), out_test.numpy().ravel())**.5}
        
def deployment_GB_1d(
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None
        ):
    """
    TBA

    Parameters
    ----------
    index : TYPE, optional
        DESCRIPTION. The default is '^GSPC'.
    start_date : TYPE, optional
        DESCRIPTION. The default is '2000-01-01'.
    lag : TYPE, optional
        DESCRIPTION. The default is 20.
    include_rv : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    
    prices = yf.Ticker(index).history(start = start_date).Close
    rets = 100*(prices.pct_change().dropna())
    rets_train, rets_test = train_test_split_ts(rets, .7)
    
    X_train, y_train = take_X_y(rets_train, lag, take_rv = include_rv, log_rv =include_rv, reshape = True )
    X_test, y_test = take_X_y(rets_test, lag, take_rv = include_rv, log_rv =include_rv, reshape = True)
    lgb_train, lgb_test = lgb.Dataset(X_train, y_train, free_raw_data=False ),\
    lgb.Dataset(X_test, y_test,  free_raw_data=False )
    
    lgbm_params = {
        'max_depth':1,
        'learning_rate' : .2,
        'boosting':'gbdt',
        'num_iterations':200,
        'force_col_wise ':'true',
        'early_stopping_round':10,
        'tree_learner': 'serial' ,
        'bagging_fraction': 1,
        'feature_fraction': 1,
        'extra_trees':'true'
    }
    
    model = lgb.train(
        params = lgbm_params,
        train_set = lgb_train,
        valid_sets = lgb_test,
        fobj  = nll_gb_exp,
        feval = nll_gb_exp_eval,
        verbose_eval = False
    )
    
    print(60*'*')
    print(pyfiglet.figlet_format("             MODEL\nEVALUATION"))
    print(60*'*')
    
    out = model.predict(X_train)
    plt.plot(np.exp(out)**.5)
    plt.title('Gradient Boosting')
    plt.show()
    
    print('\n\nPerformance on the Train Set:\n'+30*'-'+'\n')

    print('GB NLL: {:6.0f}'.format(nll_gb_exp_eval(out, lgb_train)[1]))
    
    garch = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'GARCH', p=1, q=1)
    fit = garch.fit(disp = False)
    g_vol = fit.conditional_volatility
    
    plt.plot(g_vol)
    plt.title('GARCH')
    plt.show()
    print('Garch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol**2), lgb_train)[1]))
    
    #TODO: add egarch and gjr 
    
    print('\n\nPerformance on the Test Set:\n'+30*'-'+'\n')
    plt.plot(np.exp(model.predict(X_test))**.5, label = 'GB')
    plt.plot(np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatilty')
    plt.title('Gradient Boosting Out of Sample '+ index)
    plt.legend()
    if save!=None:
        plt.savefig(save+'\\GB__'+index+'.png')
    plt.show()


    print('GB NLL: {:6.0f}'.format(nll_gb_exp_eval(model.predict(X_test), lgb_test)[1]))
    print('GB RMSE: {:1.3f}'.format(mse(np.exp(model.predict(X_test))**.5, np.exp(X_test[:,-1]))**.5))
    
    
    g_vola_pred = forward_garch(tf.convert_to_tensor(y_test), tf.convert_to_tensor(y_train), fit).numpy().ravel()
    plt.plot(g_vola_pred, label = 'GARCH')
    plt.plot( np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatilty')
    plt.legend()
    plt.title('GARCH Out of Sample '+ index)
    if save!=None:
        plt.savefig(save+'\\GARCH__'+index+'.png')
        plt.show()


    print('Garch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vola_pred**2), lgb_test)[1]))
    print('Garch RMSE: {:1.3f}'.format(mse(g_vola_pred, np.exp(X_test[:,-1]))**.5))
    
    #TODO: add egarch and gjr
    
    return {'name': 'GB__'+index,
            'NLL': nll_gb_exp_eval(model.predict(X_test), lgb_test)[1],
            'RMSE': mse(np.exp(model.predict(X_test))**.5, np.exp(X_test[:,-1]))**.5,
            'GARCH NLL': nll_gb_exp_eval(np.log(g_vola_pred**2), lgb_test)[1],
            'GARCH RMSE': mse(g_vola_pred, np.exp(X_test[:,-1]))**.5
            #TODO: add egarch and gjr
            }

    

#end\
    
def output1(output_file =  r'C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Figures\1d'):
    """
    Generation of results and figures for the One-dimensional case.
    
    List of Tickers used: ['^GSPC', '^DJI', '^IXIC', '^RUT', '^SSMI', '^OEX', '^N225', '^FTSE']

    Returns
    -------
    None.

    """
    
    tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^SSMI', '^OEX', '^N225', '^FTSE']
    nll_results = []
    rmse_results = []
    
    for t in tqdm(tickers):
        nll = []
        rmse = []
        print(t)
        gb = deployment_GB_1d(index = t, save = output_file)
        rnn = deployment_RNN_1d(lstm = False, index = t, save = output_file)
        lstm = deployment_RNN_1d(lstm = True, index = t, save = output_file)
        fnn = deployment_DNN_1d(index = t, save = output_file)
        nll.append([t, rnn['NLL'].numpy()[0], lstm['NLL'].numpy()[0],
                    fnn['NLL'].numpy()[0], gb['NLL'], gb['GARCH NLL']])
        rmse.append([t, rnn['RMSE'], lstm['RMSE'], fnn['RMSE'],
                     gb['RMSE'], gb['GARCH RMSE']])
        nll_results.append(nll)
        rmse_results.append(rmse)

    return nll_results, rmse_results
    
