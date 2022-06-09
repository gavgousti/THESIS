# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:00:44 2022

@author: Giorgio
"""



import pandas as pd
from tensorflow.keras.utils import timeseries_dataset_from_array as loader
import tensorflow as tf, numpy as np, seaborn as sns
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from UTILS import take_X_y, train_test_split_ts
from thesis import CC
from tensorflow import keras
from keras.layers import Flatten, Dense, LSTM, Reshape, SimpleRNN, BatchNormalization
from keras.initializers import RandomNormal

# =============================================================================
# 
# =============================================================================

def get_cov(X, lamda = 1e-3):
    cov = tf.convert_to_tensor(
        [M@tf.transpose(M)+lamda*tf.eye(M.shape[0]) for M in X], dtype = 'float32'
        )
    return cov

def tf_nll_multi(X, mu, lamda=1e-3):
    start = time.time()
    nll = 0
    for t in range(mu.shape[0]):
        cov = X[t]@tf.transpose(X[t])+lamda*tf.eye(X[t].shape[0])
        nll+=.5*(tf.math.log(tf.linalg.det(cov))+\
                tf.expand_dims(mu[t], 0)@\
                    tf.linalg.inv(cov)@\
                        tf.expand_dims(mu[t], 1))
    print('\n\n\n Passed the loss fn in {:0.5f} seconds!\n\n\n'.format(time.time()-start))
    return nll



def take_X_y_MD(prices, lag = 2, get_outers = False):
    rets = 100*prices.pct_change().dropna()
    
    if get_outers:
        outers = np.array([
            (rets.iloc[i, :].values.reshape(-1,1)@\
             rets.iloc[i, :].values.reshape(-1,1).transpose()).ravel().tolist()\
                for i in range(rets.shape[0])
                ])
    
        load = loader(
                data = outers,
                targets = rets.iloc[lag:],
                sequence_length=lag,
                batch_size=rets.shape[0]
                )
    else:
        load = loader(
                data = rets,
                targets = rets.iloc[lag:],
                sequence_length=lag,
                batch_size=rets.shape[0]
                )

    for X, y in load:
        X, y = tf.cast(X, 'float32'), tf.cast(y, 'float32')
    
    return X, y


class MultiNet(keras.Model):
    
    def __init__(
            self,
            hidden_size= 100,
            activations = ['tanh', 'exponential'],
            d = 19,
            lstm =True
            ):
        super().__init__(self)
        self.hidden_size = hidden_size
        self.activations = activations
        self.d = d; self.out = int(.5*(d**2-d)+d)
        self.__layers = [
            BatchNormalization(),
            LSTM(self.hidden_size, activation = self.activations[0]),
            BatchNormalization(),
            Dense(self.d**2, activation=self.activations[1]),
            Reshape((self.d, self.d))
            ]
        if not lstm:
            self.__layers[1] = SimpleRNN(self.hidden_size, activation = self.activations[0])


    
    def call(self, X):
        for f in self.__layers:
            X = f(X)    
        return X
    
    def train(
            self,
            X,
            y,
            lr,
            epochs,
            bs,
            lamda_get_cov
            ):
        self.loss_train = []
        self.compile(optimizer = tf.keras.optimizers.Adam(lr),
                      loss = tf_nll_multi)
        for epoch in tqdm(range(1, epochs+1)):
            for i in range(X.shape[0]//bs+1):
                
                X_, y_ = X[i*bs: i*bs+bs], y[i*bs: i*bs+bs]
                if i ==X.shape[0]//bs:
                    if X.shape[0]==bs:
                        break
                    X_, y_ = X[i*bs:X.shape[0]], y[i*bs:y.shape[0]]

                with tf.GradientTape() as tape:
                    logits = self(X_)
                    loss = tf_nll_multi(logits, y_, lamda = lamda_get_cov)
                    
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients,
                                                    self.trainable_weights))
                self.loss_train.append(loss[0,0])
           
            if epoch%5==0:
                print('\nEPOCH: {}\n'.\
            format(epoch)+30*'-'+'\nTRAIN LOSS: {:5.0f}'\
            .format(tf_nll_multi(self(X), y, lamda_get_cov).numpy()[0,0]))
                    
                print(30*'=')

# =============================================================================
# 
# =============================================================================


path = r"C:\Users\gavgous\OneDrive - Zanders-BV\Desktop\THESIS\GITHUB\THESIS-main\DATA"
data = pd.read_csv(path+"\\"+'SMI_data.csv', index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

train, test = train_test_split_ts(data)

X, y = take_X_y_MD(train, lag = 20, get_outers=True) 
X_test, y_test = take_X_y_MD(test, lag = 20, get_outers=True) 

model = \
MultiNet(hidden_size=5, activations = ['tanh', 'linear'], d =train.shape[1], lstm = True)

model(X)    
model.summary() 
model.train(X, y, lr=1e-3, epochs=30, bs=X.shape[0], lamda_get_cov=1e-1)

plt.plot(get_cov(model(X), 1e-1).numpy().sum(1).sum(1))
