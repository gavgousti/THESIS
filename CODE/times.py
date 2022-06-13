# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:08:16 2022

@author: Giorgio
"""

import pickle, pandas
from thesis import deployment_DNN_1d, deployment_GB_1d, deployment_RNN_1d
from tqdm import tqdm

with open(r"C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Data\1d_returns.pkl", 'rb') as f:
    data = pickle.load(f)

times = []
for t in tqdm(range(len(data))):
    gb = deployment_GB_1d(save = None, returns_file = True,file=data[t])
    
    rnn = deployment_RNN_1d(lstm = False, save = None, returns_file = True,file=data[t])
    
    lstm = deployment_RNN_1d(lstm = True, save = None, returns_file = True,file=data[t])
    
    fnn = deployment_DNN_1d(save = None, returns_file = True,file=data[t])
    
    times.append([rnn['TIME'], lstm['TIME'], fnn['TIME'], gb['TIME GB'],
                  gb['TIME GARCH'], gb['TIME GJR'], gb['TIME EGARCH']])
    print(times)
    
table = pandas.DataFrame(times, columns = ['RNN', 'LSTM', 'FNN', 'GB', 'GARCH',
                                         'GJR', 'EGARCH'],
                         index = [stock.name[1:] for stock in data]).transpose()

table.to_csv(r"C:\Users\Giorgio\Desktop\Master\THESIS CODES ETC\Tables\times_1d.csv")
