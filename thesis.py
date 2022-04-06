# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:51 2022

@author: Giorgio
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, LSTM
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib as mpl
from UTILS import nll


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