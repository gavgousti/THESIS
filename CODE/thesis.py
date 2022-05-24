# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:51 2022

@author: Giorgio
"""

from tensorflow import keras
from keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, LSTM
from tqdm import tqdm
import matplotlib as mpl, time
from scipy.optimize import minimize
import numpy as np, tensorflow as tf, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from UTILS import train_test_split_ts, forward_garch, nll_gb_exp,\
 nll_gb_exp_eval, take_X_y,nll, forward_gjr, forward_egarch, dcc_nll,\
     take_DCC_cov, forward_CC
import plotly.graph_objects as go
from plotly.offline import plot
import pyfiglet
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
from arch import arch_model

mpl.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')

class CC():

    def __init__(
            self,
            prices, 
            scale = 100,
            correlation_type = 'Constant'
            ):
        '''
        Conditional Correlation Model

        Parameters
        ----------
        prices : pd.DataFrame
            The multidimensional time series of the assets' prices
            shape = (T, d)
        scale : int, optional
            Scalling of the returns. The default is 100.
        correlation_type : str, optional
            Type of the correation structure.
            Available types "Constant" or "Dynamic"
            The default is 'Constant'.

        Returns
        -------
        None.

        '''
        self.data = prices
        self.returns =scale*prices.pct_change().dropna()
        self.returns -= self.returns.mean()
        self.garch_models  = {}
        self.stock_names = self.returns.columns
        self.covariances = []
        self.volatilities = self.returns.copy()-self.returns.copy()
        self.type = correlation_type
        self.real_covariance = []
        
        for i in range(22, self.returns.shape[0]):
            self.real_covariance.append(self.returns.iloc[i-22:i, :].cov().\
                                        values)
        self.real_covariance = np.array(self.real_covariance)

    
    def fit(
            self,
            volatility = 'GARCH'
            ):
        
        #TODO: Extend for more volatility inputs
        
        print('\nFitting '+self.type+' CC Model...\n'+30*'‾'+'\n')
        
        print('\n• Fitting individual volatility models...')
        if volatility == 'GARCH':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = arch_model(self.returns[stock],
                                                  mean = 'Constant',
                                                  vol = 'GARCH',
                                                  p=1,
                                                  q=1).fit(disp = False)
                self.volatilities[stock]=\
                    self.garch_models[stock].conditional_volatility.values
       
        self.P_c = np.divide(self.returns, self.volatilities).cov()
        
        if self.type == 'Constant':
            self.theta = (0,0)
            print('\n• Calculating conditional covariance matrices...')
            for t in tqdm(range(self.returns.shape[0])):
                Delta = np.diag(self.volatilities.iloc[t])
                cov = Delta@self.P_c@Delta
                self.covariances.append(cov)
                del cov
            self.covariances = np.array(self.covariances)
                
            print('\n• Calculating NLL...')
            self.nll = dcc_nll((0,0), self.volatilities, self.returns)
            print('NLL: {:6.0f}'.format(self.nll))
        
        elif self.type == 'Dynamic':
            start = time.time()
            print('\n• Optimization for the parameters alpha and beta...\n')
            #TODO: check if we can add the jacobian to save time
            self.nll_optimization = minimize(
                fun =  dcc_nll,
                x0 = (.05, .05),
                args = (self.volatilities, self.returns),
                method = 'SLSQP',
                bounds = [(0,1), (0,1)],
                constraints = {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]},
                options = {'maxiter': 30, 'disp': True}    
                )
            theta = self.nll_optimization.x
            self.theta = theta
            end = time.time()
            print('\n• Calculating conditional covariance matrices...')
            self.covariances = take_DCC_cov(
                theta,
                self.volatilities,
                self.P_c,
                self.returns
                )

            print('\n   ➼Time Elapsed for Optimization: {:3.0f}"'\
                  .format(end-start))
            
            self.nll = dcc_nll(theta,
                               self.volatilities,
                               self.returns
                               )

            print('   ➼NLL Value: {:7.0f}'.format(self.nll))
            print('   ➼alpha: {:1.3f} | beta: {:1.3f}\n'.\
                  format(theta[0], theta[1]))
        
        plt.plot(self.covariances.sum(1).sum(1)[22:]**.5,
                 label = 'Model Volatility')
        plt.plot(self.real_covariance.sum(1).sum(1)**.5,
                 label = 'Realized Volatility')
        plt.legend()
        self.rmse_train = mse(self.covariances.sum(1).sum(1)[22:]**.5,
                              self.real_covariance.sum(1).sum(1)**.5)**.5
        \
    plt.title('Outputted Volatility Comparison on Train Set (RMSE: {:3.3f})'.\
                  format(self.rmse_train))
        plt.show()
        

            
    def visualize_cov(
            self
            ):
        '''
        Visualization of Time Varying Cov Matrix.
        Returns
        -------
        None.
        '''
        for t in tqdm(range(0, self.returns.shape[0], 30)):
            df = pd.DataFrame(self.covariances[t, :, :])
            df.columns, df.index = self.returns.columns, self.returns.columns
            sns.heatmap(df, cmap = 'viridis')
            plt.title(self.returns.index[t])
            plt.show()
    
    def check_pd(self):
        '''
        Check if every covariance matrix is positive definite
        Returns
        -------
        bool
        '''
        cond = []
        for t in tqdm(range(self.covariances.shape[0])):
            cond.append(np.all(np.linalg.eigvals(self.covariances[t, :, :])>0))
        return np.all(cond)
    
    def visualize_loss_fn(
            self
            ):
        if self.type == 'Constant':
            print('!!! Only available for Dynamic Correlation !!!')
        else:
            N = 25
            values = []
            alpha_grid = np.linspace(0, .999, N)
            beta_grid = np.linspace(0, .99, N)
            for alpha in tqdm(alpha_grid):
                tempo = []
                for beta in beta_grid:
                    if beta+alpha<.99:
                        val = dcc_nll((alpha, beta),
                                      self.volatilities,
                                      self.returns
                                      )
                        if val<2e5:
                            tempo.append(val)
                        else:
                            tempo.append(np.nan)
                    else:
                        tempo.append(np.nan)
                values.append(tempo)
            values = pd.DataFrame(values,
                                  columns = beta_grid,
                                  index = alpha_grid)

            sns.heatmap(values, vmax=None, cmap='viridis')
            plt.show()

            fig = go.Figure(data=[go.Surface(colorscale='Viridis',
                                             z=values.values,
                                             x = beta_grid,
                                             y = alpha_grid)])
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
            
    def evaluation_on_test_set(
            self,
            test
            ):
        '''
        Evaluation on a test set.

        Parameters
        ----------
        test : pd.DataFrame
            PRICES on the test set.

        Returns
        -------
        dict
        collection of i-s & o-o-s rmse and nll.

        '''
        self.covariances_test, nll = forward_CC(self, test)
        self.real_covariance_test = []
        returns_test = 100*test.pct_change().dropna()
        for i in range(22, returns_test.shape[0]):
            self.real_covariance_test.append(returns_test\
                                             .iloc[i-22:i, :].cov().values)
        self.real_covariance_test = np.array(self.real_covariance_test)
        
        plt.plot(self.covariances_test.sum(1).sum(1)[22:]**.5,
                 label = 'Model Volatility')
        plt.plot(self.real_covariance_test.sum(1).sum(1)**.5,
                 label = 'Realized Volatility')
        plt.legend()
        rmse = mse(self.covariances_test.sum(1).sum(1)[22:]**.5,
                   self.real_covariance_test.sum(1).sum(1)**.5)**.5
        \
    plt.title('Outputted Volatility Comparison on Test Set (RMSE: {:3.3f})'.\
                  format(rmse))
        plt.show()
        return {
            'RMSE TRAIN SET': self.rmse_train,
            'NLL TRAIN SET': self.nll,
            'RMSE TEST SET': rmse,
            'NLL TEST SET': nll
            }
        
            
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
    
# =============================================================================
#     GARCH
# =============================================================================
    
    garch = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'GARCH', p=1, q=1)
    fit = garch.fit(disp = False)
    g_vol = fit.conditional_volatility
    
    plt.plot(g_vol)
    plt.title('GARCH')
    plt.show()
    print('Garch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol**2), lgb_train)[1]))
    
# =============================================================================
#     GJR
# =============================================================================
    gjr = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'GARCH', p=1, q=1, o=1)
    fit_gjr = gjr.fit(disp = False)
    g_vol_gjr= fit_gjr.conditional_volatility
    
    plt.plot(g_vol_gjr)
    plt.title('GJR')
    plt.show()
    print('Gjr  NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol_gjr**2), lgb_train)[1]))
    
# =============================================================================
#     EGARCH
# =============================================================================
    egarch = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'EGARCH', p=1, q=1, o=1)
    fit_egarch = egarch.fit(disp = False)
    g_vol_egarch = fit_egarch.conditional_volatility
    
    plt.plot(g_vol_egarch)
    plt.title('EGARCH')
    plt.show()
    print('Egarch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol_egarch**2), lgb_train)[1]))


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
    
# =============================================================================
#     GARCH
# =============================================================================
    
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
    
# =============================================================================
#     GJR
# =============================================================================
    
    g_vola_pred_gjr = forward_gjr(tf.convert_to_tensor(y_test), tf.convert_to_tensor(y_train), fit_gjr).numpy().ravel()
    plt.plot(g_vola_pred_gjr, label = 'GJR')
    plt.plot(np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatilty')
    plt.legend()
    plt.title('GJR Out of Sample '+ index)
    if save!=None:
        plt.savefig(save+'\\GJR__'+index+'.png')
        plt.show()


    print('Gjr NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vola_pred_gjr**2), lgb_test)[1]))
    print('Gjr RMSE: {:1.3f}'.format(mse(g_vola_pred_gjr, np.exp(X_test[:,-1]))**.5))
    
# =============================================================================
#     EGARCH
# =============================================================================
    
    g_vola_pred_egarch = forward_egarch(tf.convert_to_tensor(y_test), tf.convert_to_tensor(y_train), fit_egarch).numpy().ravel()
    plt.plot(g_vola_pred_egarch, label = 'EGARCH')
    plt.plot(np.exp(X_test[:,-1]), alpha = 1, label = 'Realized Volatilty')
    plt.legend()
    plt.title('EGARCH Out of Sample '+ index)
    if save!=None:
        plt.savefig(save+'\\EGARCH__'+index+'.png')
        plt.show()


    print('Egarch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vola_pred_egarch**2), lgb_test)[1]))
    print('Egarch RMSE: {:1.3f}'.format(mse(g_vola_pred_egarch, np.exp(X_test[:,-1]))**.5))
    
    return {'name': 'GB__'+index,
            'NLL': nll_gb_exp_eval(model.predict(X_test), lgb_test)[1],
            'RMSE': mse(np.exp(model.predict(X_test))**.5, np.exp(X_test[:,-1]))**.5,
            'GARCH NLL': nll_gb_exp_eval(np.log(g_vola_pred**2), lgb_test)[1],
            'GARCH RMSE': mse(g_vola_pred, np.exp(X_test[:,-1]))**.5,
            'GJR NLL': nll_gb_exp_eval(np.log(g_vola_pred_gjr**2), lgb_test)[1],
            'GJR RMSE': mse(g_vola_pred_gjr, np.exp(X_test[:,-1]))**.5,
            'EGARCH NLL': nll_gb_exp_eval(np.log(g_vola_pred_egarch**2), lgb_test)[1],
            'EGARCH RMSE': mse(g_vola_pred_egarch, np.exp(X_test[:,-1]))**.5
            }

    
    
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
        nll.append([t, rnn['NLL'].numpy(), lstm['NLL'].numpy(),
                    fnn['NLL'].numpy(), gb['NLL'], gb['GARCH NLL'],
                    gb['GJR NLL'], gb['EGARCH NLL']])
        rmse.append([t, rnn['RMSE'], lstm['RMSE'], fnn['RMSE'],
                     gb['RMSE'], gb['GARCH RMSE'],
                     gb['GJR RMSE'], gb['EGARCH RMSE']])
        nll_results.append(nll)
        rmse_results.append(rmse)

    return nll_results, rmse_results
