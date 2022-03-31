# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:51 2022

@author: Giorgio
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib as mpl
from arch import arch_model
from arch.univariate import GARCH
import tensorflow_probability as tfp
import seaborn as sns
import tensorflow.math as m
from sklearn.metrics import mean_squared_error as mse
from UTILS import nll


mpl.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')

class RNN(keras.Model):
    def __init__(self,
                 hiddens = [100, 1],
                 activation = 'relu',
                 optimizer= 'adam',
                 last_activation = 'relu',
                 drop_rate = .6
                ):
        super().__init__()
        self.optimizer = optimizer
        self.drop_rate = drop_rate
        self.dropout = Dropout(self.drop_rate)
        self.activation = activation; self.last_activation = last_activation
        self.hiddens = hiddens
        self.initializer =  tf.keras.initializers.RandomNormal(mean=0.0, stddev=30)
        self.rnn = SimpleRNN(self.hiddens[0],
                             activation = self.activation,
                             kernel_initializer = self.initializer                         
                            )
        self.denses = []
        for size in self.hiddens[1:-1]:
            self.denses.append(Dense(size, activation = self.activation))
        self.denses.append(Dense(self.hiddens[-1], activation = self.last_activation))
        
    def call(self, x):
        out = self.rnn(x)
        for l in self.denses:
            out = l(out)
            out = self.dropout(out)
            return out
    
    def train(self,
              x,
              epochs = 10,
              loss_fn = nll,
              use_rv = False,
              scale = 1
             ):
        #see if batched data is needed
#         x[:,0,:] = x[:,0,:]*scale
        print(30*'*'+'\nFITING THE MODEL\n'+30*'*')
        self.loss_to_plot = []
        self.compile(optimizer = self.optimizer, loss = loss_fn)
        for epoch in tqdm(range(epochs)):
            with tf.GradientTape() as tape:
                logits = self(x)**.5
                loss = loss_fn(logits, x[:,0,:])
            print(30*'-')
            print('EPOCH:{}'.format(epoch+1))
            print('NLL:{}'.format(loss))
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_to_plot.append(loss.numpy())
    
    def plot_loss(self):
        plt.plot(self.loss_to_plot)


stock = yf.Ticker('^GSPC').history(start = '2010-01-01', end = '2022-03-01').Close
rets_ = stock.pct_change().dropna()
rets = rets_-rets_.mean()
tf_rets = tf.convert_to_tensor(rets.values,  dtype='float32')[21:]
tf_rv = tf.convert_to_tensor(rets.rolling(22).std().dropna().values, dtype = 'float32')
#data = tf.stack((tf_rets, tf_rv), axis= 1)
data = tf.reshape(tf_rets, (tf_rets.shape[0],1, 1))
# tf_rets = tf.reshape(tf_rets, (tf_rets.shape[0],1,1))


model = RNN(hiddens = [100,1], activation='tanh', last_activation = 'sigmoid', drop_rate = 0)
model(data)
model.summary()
model.train(data, epochs = 300, use_rv = False, loss_fn = nll)


garch = arch_model(rets, mean='Zero', vol='GARCH', p=1, q=1)
fit = garch.fit()

sns.distplot((model(data)**.5).numpy().ravel(), label = 'model')
sns.distplot(fit.conditional_volatility.values, label = 'garch')
sns.distplot(tf_rv.numpy().ravel(), label = 'rv')
plt.legend()

model.plot_loss()

#plt.plot(rets.iloc[21:], label = 'returns')
plt.plot(rets.iloc[21:].index, model(data)**.5, label = 'model ({:7.2f})'.format(-model.loss_to_plot[-1]))
plt.plot(rets.iloc[21:].index, fit.conditional_volatility.iloc[21:],
         label = 'garch ({:7.2f})'.format(fit.loglikelihood))
plt.plot(rets.iloc[21:].index,tf_rv.numpy().ravel(), label = 'rv')
#take a look on what you plot on garch nll later
plt.plot()
plt.text(x = rets.iloc[21:].index[10], y = 0.8*max(model(data)**.5), s = 'GARCH VS RV RMSE:{:1.5f}'\
         .format(mse(tf_rv.numpy().ravel(),fit.conditional_volatility.iloc[21:]) **.5),
         fontsize = 'xx-large')
plt.text(x = rets.iloc[21:].index[10], y = .7*max(model(data)**.5), s = 'MODEL VS RV RMSE:{:1.5f}'\
         .format(mse(tf_rv.numpy().ravel(), model(data)**.5)**.5), fontsize = 'xx-large')
plt.legend()
plt.show()