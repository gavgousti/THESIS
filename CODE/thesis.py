# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:51 2022

@author: Giorgio
"""
#TODO: docstrings

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
     take_DCC_cov, forward_CC, simulate_garch, cov_to_corr
import plotly.graph_objects as go
from plotly.offline import plot
import pyfiglet
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
from arch import arch_model
from time import time as now
import scipy; Phi = scipy.stats.norm.cdf

mpl.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')

class GB():
    
    def __init__(
            self,
            lgbm_params={
                'max_depth':1,
                'learning_rate' : .2,
                'boosting':'gbdt',
                'num_iterations':50,
                'force_col_wise ':'true',
                'early_stopping_round':10,
                'tree_learner': 'serial' ,
                'bagging_fraction': 1,
                'feature_fraction': 1,
                'extra_trees':'true'
                },
            scale = 100,
            lag = 20
            ):
        self.lgbm_params = lgbm_params
        self.lag = lag
        
    def fit(
            self,
            rets_train,
            take_rv = True,
            log_rv =True,
            reshape = True
            ):
        
        X_train, y_train = take_X_y(
            rets_train,
            self.lag,
            take_rv = take_rv,
            log_rv =log_rv,
            reshape = reshape
            )
        
        lgb_train = lgb.Dataset(
            X_train,
            y_train,
            free_raw_data=False
            )
        
        self.model = lgb.train(
            params = self.lgbm_params,
            train_set = lgb_train,
            valid_sets = lgb_train,
            feval = nll_gb_exp_eval,
            fobj  = nll_gb_exp,
            verbose_eval = False
        )
        
    def predict(
            self,
            rets_test,
            take_rv = True,
            log_rv =True,
            reshape = True
            ):
        X_test, y_test = take_X_y(
            rets_test,
            self.lag,
            take_rv = take_rv,
            log_rv =log_rv,
            reshape = reshape
            )
        out = np.exp(self.model.predict(X_test))**.5
        out_pd = rets_test.copy()
        out_pd.iloc[-out.shape[0]:] = out
        out_pd = out_pd.iloc[-out.shape[0]:]
        del out
        return out_pd
 
class CC():

    def __init__(
            self,
            prices, 
            scale = 100,
            correlation_type = 'Constant',
            volatility = 'GARCH'
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
        self.vola_type = volatility
        self.data = prices
        self.returns =scale*prices.pct_change().dropna()
        self.returns -= self.returns.mean()
        self.garch_models  = {}
        self.stock_names = self.returns.columns
        self.covariances = []
        self.volatilities = self.returns.copy()-self.returns.copy()
        self.volatilities = self.volatilities.iloc[41:]
        self.type = correlation_type
        self.real_covariance = []
        
        for i in range(22, self.returns.shape[0]):
            self.real_covariance.append(self.returns.iloc[i-22:i, :].cov().\
                                        values)
        self.real_covariance = np.array(self.real_covariance)

    
    def fit(
            self,
            ):
        '''
        Fitting the Constant Correlation model to data:
            1. Individual conditional volatilities estimation.
            2. Estimation of the conditional covariance matrices.

        Parameters
        ----------
        volatility : str, optional
            Type of model for individual volatilities. The default is 'GARCH'.
            So far, available:
                i) GARCH
                ii) EGARCH
                iii) GJR
                iv) GB
                v) FNN
                vi) LSTM
                

        Returns
        -------
        None.

        '''
                
        print('\nFitting '+self.type+' CC Model...\n'+30*'‾'+'\n')
        
        print('\n• Fitting individual volatility models...')
        if self.vola_type == 'GARCH':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = arch_model(self.returns[stock],
                                                  mean = 'Constant',
                                                  vol = 'GARCH',
                                                  p=1,
                                                  q=1).fit(disp = False)
                self.volatilities[stock]=\
                    self.garch_models[stock].conditional_volatility.values[41:]
        elif self.vola_type == 'EGARCH':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = arch_model(self.returns[stock],
                                                  mean = 'Constant',
                                                  vol = 'EGARCH',
                                                  p=1,
                                                  q=1,
                                                  o=1).fit(disp = False)
                self.volatilities[stock]=\
                    self.garch_models[stock].conditional_volatility.values[41:]
        elif self.vola_type == 'GJR':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = arch_model(self.returns[stock],
                                                  mean = 'Constant',
                                                  vol = 'GARCH',
                                                  p=1,
                                                  q=1,
                                                  o=1).fit(disp = False)
                self.volatilities[stock]=\
                    self.garch_models[stock].conditional_volatility.values[41:]     
        elif self.vola_type == 'GB':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = GB()
                self.garch_models[stock].fit(self.returns[stock])
                self.volatilities[stock]=\
                self.garch_models[stock].predict(self.returns[stock]).values
                
        elif self.vola_type == 'FNN':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = DNN(
                    hidden_size = [300,],
                    dropout = .5,
                    l1 = 1,
                    l2 = 1
                    )
                lag =20
                X_train, y_train = take_X_y(self.returns[stock],
                                            lag,
                                            reshape = True,
                                            take_rv = True,
                                            log_rv =True)

                X_train, y_train = tf.convert_to_tensor(
                    X_train,
                    dtype  = 'float32'
                    ),\
                    tf.convert_to_tensor(
                        y_train,
                        dtype  = 'float32'
                        )
                                        
                self.garch_models[stock].train(
                    X_train,
                    y_train,
                    X_train,
                    y_train,
                    epochs = 20,
                    bs = 1024,
                    lr = 1e-3
                    )
                self.volatilities[stock]=\
                    self.garch_models[stock](X_train).numpy().ravel()
                    
        elif self.vola_type == 'LSTM':
            for stock in tqdm(self.returns.columns):
                self.garch_models[stock] = RNN(
                    lstm = True,
                    hidden_size=[60,],
                    l1 = 0                  
                    )
                lag =20
                X_train, y_train = take_X_y(self.returns[stock],
                                            lag,
                                            reshape = False,
                                            take_rv = True,
                                            log_rv =True)

                X_train, y_train = tf.convert_to_tensor(
                    X_train,
                    dtype  = 'float32'
                    ),\
                    tf.convert_to_tensor(
                        y_train,
                        dtype  = 'float32'
                        )
                    
                self.garch_models[stock].train(
                    X_train, 
                    y_train,
                    X_train,
                    y_train,
                    epochs = 10,
                    bs = 512,
                    lr = .008
                )
                
                self.volatilities[stock]=\
                    self.garch_models[stock](X_train).numpy().ravel()
                    
        self.P_c = np.divide(self.returns.iloc[41:],
                             self.volatilities).cov()
        
        if self.type == 'Constant':
            self.theta = (0,0)
            print('\n• Calculating conditional covariance matrices...')
            for t in tqdm(range(41, self.returns.shape[0])):
                Delta = np.diag(self.volatilities.iloc[t-41])
                cov = Delta@self.P_c@Delta
                self.covariances.append(cov)
                del cov
            self.covariances = np.array(self.covariances)
                
            print('\n• Calculating NLL...')
            self.nll = dcc_nll((0,0), self.volatilities, self.returns.\
                               iloc[41:])
            print('NLL: {:6.0f}'.format(self.nll))
        
        elif self.type == 'Dynamic':
            start = time.time()
            print('\n• Optimization for the parameters alpha and beta...\n')
            #TODO: check if we can add the jacobian to save time
            self.nll_optimization = minimize(
                fun =  dcc_nll,
                x0 = (.05, .05),
                args = (self.volatilities, self.returns.iloc[41:]),
                method = 'SLSQP',
                bounds = [(0,1), (0,1)],
                constraints = {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]},
                options = {'maxiter': 60, 'disp': True}    
                )
            theta = self.nll_optimization.x
            self.theta = theta
            end = time.time()
            print('\n• Calculating conditional covariance matrices...')
            self.covariances, self.Q_t= take_DCC_cov(
                theta,
                self.volatilities,
                self.P_c,
                self.returns.iloc[41:]
                )

            print('\n   ➼Time Elapsed for Optimization: {:3.0f}"'\
                  .format(end-start))
            
            self.nll = dcc_nll(theta,
                               self.volatilities,
                               self.returns.iloc[41:]
                               )

            print('   ➼NLL Value: {:7.0f}'.format(self.nll))
            print('   ➼alpha: {:1.3f} | beta: {:1.3f}\n'.\
                  format(theta[0], theta[1]))
        
        # plt.plot(self.covariances.sum(1).sum(1)[22:]**.5,
        #          label = 'Model Volatility')
        # plt.plot(self.real_covariance[41:].sum(1).sum(1)**.5,
        #          label = 'Realized Volatility')
        # plt.legend()
        self.rmse_train = mse(self.covariances.sum(1).sum(1)[22:]**.5,
                              self.real_covariance[41:].sum(1).sum(1)**.5)**.5
    #     \
    # plt.title('{} CC - {} | Evaluation on Train Set (RMSE: {:3.3f})'.\
    #               format(self.type, self.vola_type, self.rmse_train))
    #     plt.show()
        

            
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
                                      self.returns.iloc[41:]
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
        self.real_covariance_test = np.array(self.real_covariance_test)[41:]
        
        # plt.plot(self.covariances_test.sum(1).sum(1)[22:]**.5,
        #          label = 'Model Volatility')
        # plt.plot(self.real_covariance_test.sum(1).sum(1)**.5,
        #          label = 'Realized Volatility')
        # plt.legend()
        rmse = mse(self.covariances_test.sum(1).sum(1)[22:]**.5,
                   self.real_covariance_test.sum(1).sum(1)**.5)**.5
        
        # plt.title('{} CC - {} | Evaluation on Test Set (RMSE: {:3.3f})'.\
        #               format(self.type, self.vola_type, rmse))

        # plt.show()
        return {
            'MODEL': self.type+' CC - '+self.vola_type,
            'RMSE TRAIN SET': self.rmse_train/len(self.stock_names),
            'NLL TRAIN SET': self.nll,
            'RMSE TEST SET': rmse/len(self.stock_names),
            'NLL TEST SET': nll,
            'alpha': self.theta[0],
            'beta': self.theta[1]
            }
    
    def DCC_GARCH_simulation(
        self,
        simulations =1000,
        horizon = 255
        ):
    
        w = np.ones((len(self.stock_names),1))
        alpha, beta = self.theta
        portf_val = len(self.stock_names)*np.ones((horizon,1))
        for _ in tqdm(range(simulations)):
        
            one_d_volatilities = []; one_d_residuals = []
            for i in range(len(self.stock_names)):
                temp = simulate_garch(self.garch_models[self.stock_names[i]].params,
                                      simulations = 1,
                                      horizon =horizon)
                one_d_volatilities.append(temp['Volatilities'].values.ravel().tolist())
                one_d_residuals.append(temp['Residuals'].values.ravel().tolist())
            one_d_volatilities = pd.DataFrame(one_d_volatilities, index = self.stock_names).transpose()
            one_d_residuals = pd.DataFrame(one_d_residuals, index = self.stock_names).transpose()
            Delta = np.diag(one_d_volatilities.iloc[0].values.ravel())
            P = self.P_c.values
            Sigma = Delta@P@Delta
            R = np.linalg.cholesky(Sigma)@np.random.randn(self.returns.shape[1],1)
            for i in range(1, horizon):
                Q = (1-alpha-beta)*self.P_c.values\
                    +alpha*one_d_residuals.iloc[i-1].values.reshape(-1,1)\
                        @one_d_residuals.iloc[i-1].values.reshape(-1,1).transpose()\
                    +beta*P
                Delta = np.diag(one_d_volatilities.iloc[i].values.ravel())
                P = cov_to_corr(Q)
                Sigma = Delta@P@Delta
                R = np.concatenate(
                    (R, np.linalg.cholesky(Sigma)@np.random.randn(self.returns.shape[1],1)),
                    1)
            S = np.cumprod(1+R/100, 1)
            portf_val = np.concatenate((
                portf_val,np.array([(w.transpose()@s.reshape(-1,1)).tolist()[0]\
                                    for s in S.transpose()])),1)
        portf_val = portf_val[:,1:]
        portf_val = pd.DataFrame(portf_val).transpose()
        return portf_val
    
    def forecast_covariance(
            self,
            horizon
            ):
        
        if self.vola_type == 'LSTM':
            pred_volas = []
            for stock in tqdm(self.stock_names):
                pred_volas.append(self.garch_models[stock].predict_1d(self.data[[stock]]).tolist())
                
            
            pred_volas = pd.DataFrame(pred_volas, index = self.stock_names).transpose()
        elif self.vola_type == 'GARCH':
            pred_volas = []
            for stock in tqdm(self.stock_names):
                pred_volas.append(
                    (self.garch_models[stock].forecast(horizon = horizon+1, reindex = False)\
                     .variance.values.ravel()**.5).tolist()
                    )
            pred_volas = pd.DataFrame(pred_volas, index = self.stock_names).transpose()
        else:
            print('Not Available for Given Volatility Type!')
            return None
        
        
        P_c = self.P_c.values
        Y_t = (self.returns.iloc[-1]/self.volatilities.iloc[-1]).values.reshape(-1,1)
        alpha, beta = self.theta
        E_P_1 = (1-alpha-beta)*P_c+ alpha*Y_t@Y_t.transpose() + beta*self.Q_t
        kappa = np.arange(1, horizon+1)
        out = []
        for k in kappa:
            out.append(((1-(alpha+beta)**(k-1))*P_c + (alpha+beta)**(k-1)*E_P_1).tolist())
        out = np.array(out)
        
        forecast_sigmas = np.array(\
        [(np.diag(pred_volas.iloc[t].values)@out[t]@np.diag(pred_volas.iloc[t].values))\
         .tolist() for t in range(horizon)])
        
        return forecast_sigmas
        
            
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
            for i in range(X_train.shape[0]//bs+1):
                X_, y_ = X_train[i*bs: i*bs+bs], y_train[i*bs: i*bs+bs]
                if i ==X_train.shape[0]//bs:
                    if X_train.shape[0]==bs:
                        break
                    X_, y_ = X_train[i*bs:X_train.shape[0]],\
                        y_train[i*bs:X_train.shape[0]]
                with tf.GradientTape() as tape:
                    logits = self(X_)
                    loss = nll((logits)**2, y_)
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients,
                                                   self.trainable_weights))
            self.loss_train.append(nll(self(X_train)**2, y_train))
            self.loss_val.append(nll(self(X_test)**2, y_test))
            if epoch%5==0:
                print('EPOCH: {}\n'.\
                      format(epoch)+30*'-'\
                          +'\nTRAIN LOSS: {:5.0f}\nTEST LOSS: {:5.0f}'\
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
            for i in range(X_train.shape[0]//bs+1):
                X_, y_ = X_train[i*bs: i*bs+bs], y_train[i*bs: i*bs+bs]
                if i ==X_train.shape[0]//bs:
                    if X_train.shape[0]==bs:
                        break
                    X_, y_ = X_train[i*bs:X_train.shape[0]], y_train[i*bs:X_train.shape[0]]
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
    
    def predict_1d(
            self,
            data,
            horizon = 5, 
            simulations = 100,
            lag = 20,
            include_rv = True
            ):
        
        rets = 100*data.pct_change().dropna()

        X_train, y_train = take_X_y(rets,
                                    lag,
                                    reshape = False,
                                    take_rv = include_rv,
                                    log_rv =include_rv)
        X_train, y_train = tf.convert_to_tensor(X_train, dtype  = 'float32'),\
                        tf.convert_to_tensor(y_train, dtype  = 'float32') 

        vola_paths = []
        for _ in tqdm(range(simulations)):
            X = X_train[-1:]
            vola_path = [self(X)[0][0].numpy()]
            
            for _ in range(horizon):
                r = np.random.randn()*vola_path[-1]
                rets = np.concatenate((X[-1][1:,0].numpy(),[r]))
                lrv = np.log(np.std(rets))
                log_r_volas = np.concatenate((X[-1][1:,1].numpy(),[lrv]))
                X = tf.convert_to_tensor(
                    np.expand_dims(
                        np.concatenate((rets.reshape(-1,1), log_r_volas.reshape(-1,1)),1),0
                        )
                    )
                vola_path.append(self(X)[0][0].numpy())
            vola_paths.append(vola_path)
        return pd.DataFrame(vola_paths).mean(0).values

def deployment_RNN_1d(
        lstm = False,
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None,
        returns_file = False,
        file = None
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
    
    if not returns_file:
        prices = yf.Ticker(index).history(start = start_date).Close
        rets = 100*(prices.pct_change().dropna())
    else:
        rets = file
        index = file.name

        
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
    
    time_ = -now()
    model.train(
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs = 10,
        bs = 2048,
        lr = .008
    )
    time_+=now()
    
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
            'RMSE': mse(np.exp(X_test[:,-1,1]), out_test.numpy().ravel())**.5,
            'TIME': time_}


def deployment_DNN_1d(
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None,
        returns_file = False,
        file = None
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
    
    if not returns_file:
        prices = yf.Ticker(index).history(start = start_date).Close
        rets = 100*(prices.pct_change().dropna())
    else:
        rets = file
        index = file.name

        
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
    
    time_ = -now()
    model.train(
        X_train, 
        y_train,
        X_test,
        y_test,
        epochs = 20,
        bs = 1024,
        lr = .001,
    )
    time_ += now()
    
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
            'RMSE': mse(np.exp(X_test[:,-1]), out_test.numpy().ravel())**.5,
            'TIME': time_}
        
def deployment_GB_1d(
        index = '^GSPC',
        start_date = '2000-01-01',
        lag = 20,
        include_rv = True,
        save = None,
        returns_file = False,
        file = None
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
    if not returns_file:
        prices = yf.Ticker(index).history(start = start_date).Close
        rets = 100*(prices.pct_change().dropna())
    else:
        rets = file
        index = file.name
        
    rets_train, rets_test = train_test_split_ts(rets, .7)

    print(rets)
    
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
    
    time_gb = -now()
    model = lgb.train(
        params = lgbm_params,
        train_set = lgb_train,
        valid_sets = lgb_test,
        fobj  = nll_gb_exp,
        feval = nll_gb_exp_eval,
        verbose_eval = False
    )
    time_gb+=now()
    
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
    time_garch = -now()
    fit = garch.fit(disp = False)
    time_garch+=now()
    g_vol = fit.conditional_volatility
    
    print(fit)
    
    plt.plot(g_vol)
    plt.title('GARCH')
    plt.show()
    print('Garch NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol**2), lgb_train)[1]))
    
# =============================================================================
#     GJR
# =============================================================================
    gjr = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'GARCH', p=1, q=1, o=1)
    time_gjr = -now()
    fit_gjr = gjr.fit(disp = False)
    time_gjr+=now()
    
    g_vol_gjr= fit_gjr.conditional_volatility
    
    print(fit_gjr)
    
    plt.plot(g_vol_gjr)
    plt.title('GJR')
    plt.show()
    print('Gjr  NLL: {:6.0f}'.format(nll_gb_exp_eval(np.log(g_vol_gjr**2), lgb_train)[1]))
    
# =============================================================================
#     EGARCH
# =============================================================================
    egarch = arch_model(lgb_train.get_label(), mean = 'Constant', vol = 'EGARCH', p=1, q=1, o=1)
    time_egarch = -now()
    fit_egarch = egarch.fit(disp = False)
    time_egarch+=now()
    g_vol_egarch = fit_egarch.conditional_volatility
    
    print(fit_egarch)
    
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
            'EGARCH RMSE': mse(g_vola_pred_egarch, np.exp(X_test[:,-1]))**.5,
            'TIME GB': time_gb, 'TIME GARCH': time_garch, 'TIME GJR': time_gjr,
            'TIME EGARCH': time_egarch
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
        
    rmse_table = pd.DataFrame([rmse_results[i][0] for i in \
                                 range(len(rmse_results))],
                 columns = ['Index', 'RNN', 'LSTM', 'FNN', \
                            'GB', 'GARCH', 'GJR', 'EGARCH'])\
        .set_index('Index')
    
    nll_table = pd.DataFrame([nll_results[i][0] for i in \
                                 range(len(nll_results))],
                 columns = ['Index', 'RNN', 'LSTM', 'FNN', \
                            'GB', 'GARCH', 'GJR', 'EGARCH'])\
        .set_index('Index')

    return nll_table, rmse_table



def breach_probability(
        dcc,
        weight = 'Uniform',
        start = 0,
        end = -1,
        plot__ = True,
        liab_ratio = .9):
    '''
    Breach Probability for a Collateral that follows a DCC process.

    Parameters
    ----------
    dcc :thesis.CC
        Fitted model of the returns.
    weight : str or np.array, optional
        Weight of the portfolio. If not Uniform then 
        array of shape (d,1). The default is 'Uniform'.
    start : int, optional
        Negative integer. How many days before last available date to start 
        analysis. The default is 0.
    end : int, optional
        Negative integer. How many days before last available date to finish
        the analysis. The default is -1.
    plot__ : bool, optional
        Decide if you need the output plotted or not. The default is True.
    liab_ratio : float, optional
        Lending value of the loan. The default is .9.

    Returns
    -------
    output : pd.DataFrame
        A time series with the collateral value, the credit limit and the 
        breach probability.

    '''
    if type(weight) == str:
        if weight == 'Uniform':
            w = dcc.data.shape[1]**(-1)*np.ones(dcc.data.shape[1]).reshape(-1,1) 
    else:
        w = weight.copy()
    e = np.ones_like(w)
    portf_val = pd.DataFrame([(w.T@dcc.data.iloc[i])[0] for i\
                              in range(dcc.data.shape[0])],
                             index = dcc.data.index,
                             columns = ['Portfolio Value'])
    
        
    time_ = dcc.data.index[-dcc.covariances.shape[0]:]
    counter = 0
    prob = []; L =[]
    for t in time_[start:-1]:
        l = liab_ratio*portf_val.loc[t].values[0]
    
        L.append(l)
        S = np.diag(dcc.data.loc[t])
    
        var = 1e-4*w.T@S@dcc.covariances[counter+1]@S@w
        prob.append(Phi((1*l-(1*w.T@S@e)[0,0])/var[0,0]))
        counter+=1
    
    output = pd.DataFrame([portf_val.iloc[-len(L):].values.ravel(), L, prob],
                 columns = portf_val.index[-len(L):],
                 index = ['Collateral Value',
                          'Credit Limit',
                          'Breach Probability']).transpose().dropna()
    
    if plot__:
        fig, axs = plt.subplots(2,1,sharex = True, tight_layout = True)
        output.iloc[start:end, :2].plot(ax = axs[0])
        output.iloc[start:end,2:].plot(ax = axs[1])
    return output

