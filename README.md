# THESIS
Code And Reading Material For Master Thesis/ Spring Semester 2022


## Comments on Results & Tunning

#### 1.Gradient Boosting 1d

- train ratio = 70%
- lag = 20
- lgbm_params = {
    'max_depth':1,
    'learning_rate' : .25,
    'boosting':'gbdt',
    'num_iterations':200,
    'force_col_wise ':'true',
    'early_stopping_round':10,
    'tree_learner': 'serial' ,
    'bagging_fraction': 1,
    'feature_fraction': 1,
    'extra_trees':'true'
}
- .history(start = '2000-01-01')

###### Results

Index  | NLL (test set)
------------- | -------------
^SSMI  | 1975 < garch
^GSPC  | 2019 < garch
^SP100 | 2055 = garch
^DJI (lag  = 10) | 2036 < garch
^FTSE | 2129 < garch

#### 2.RNN 1d

model = RNN(
    lstm = True,
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

###### Results

Index  | NLL (test set)
------------- | -------------
| |a| b
^SSMI  | 1985 | garch
^GSPC  | 2030 | garch
^SP100 |  2067 | garch
^DJI  | 2016 | garch
^FTSE |  2140 | garch
