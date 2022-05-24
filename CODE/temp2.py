# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:32:30 2022

@author: gavgous
"""

# =============================================================================
# TEMPORARAY SCRIPT
# =============================================================================


from thesis import CC
from sklearn.metrics import mean_squared_error as mse
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import time, numpy as np, matplotlib as mpl
from tqdm import tqdm
from UTILS import dcc_nll, take_DCC_cov
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.offline import plot
from arch import arch_model
cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,20)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap.colors)

path = "C:\\Users\\Giorgio\\Desktop\\Master\\THESIS CODES ETC\\Data\\"

# =============================================================================
# available data sets:
#    - FTSE_dat.csv
#    - SMI_data.csv
#    - Berkshire_Hathaway_Portfolio_data_.csv
# =============================================================================

data = pd.read_csv(path+"Berkshire_Hathaway_Portfolio_data_.csv", index_col = 'Date')
data.index = pd.DatetimeIndex(data.index)

model = CC(data, correlation_type = 'Dynamic')
model.fit()

#model.visualize_loss_fn()




