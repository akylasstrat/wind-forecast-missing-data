# -*- coding: utf-8 -*-
"""
Train models/ NYISO data with weather features/ updated code

@author: astratig
"""

import pickle
import os, sys
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
import scipy.sparse as sp
import time
import itertools
import random

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from utility_functions import * 
from QR_regressor import *
from clean_torch_custom_layers import * 
from clean_finite_adaptability_functions import * 

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['impute'] = True # If True, apply mean imputation for missing features 

    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2018-01-01' # start of train set
    params['split_date'] = '2018-06-01' # end of train set/start of test set
    params['end_date'] = '2019-01-01'# end of test set
        
    params['freq'] = '15min'    
    params['target_park'] = 'Noble Clinton'
    params['horizon'] = 16 # forecast horizon
    params['train'] = True # If True, then train models, else tries to load previous runs
    params['save'] = True # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv(f'{cd}\\data\\2018_wind_site_5min.csv', index_col = 0, parse_dates=True)
metadata_df = pd.read_csv(f'{cd}\\data\\wind_meta.csv', index_col = 0)

freq = config['freq']
target_park = config['target_park']

# min_lag: last observed measurement
# Example: Horizon=1 -> 1-step ahead forecast -> we are at time t forecasting t+1 -> min_lag == 1
min_lag = config['horizon']
print(f'Forecast horizon:{min_lag}-step ahead')
config['min_lag'] = min_lag
trained_models_path = f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps'

# ID forecasts from NREL (instead of weather)
id_forecasts = pd.read_csv(f'{cd}\\data\\Site_{target_park}_wind_intraday_2018_forecasts.csv')
time_index = pd.date_range(start='2018-01-01', end = '2019-01-01', freq = '1h')[:-1]
id_forecasts_df = pd.DataFrame(data = id_forecasts.mean(0).values, index = time_index, columns=[f'{target_park}_ID_for'])/metadata_df.loc[target_park].capacity

id_forecasts_df = id_forecasts_df.resample(freq).interpolate()

target_zone = metadata_df.loc[target_park].load_zone
power_df = power_df.resample(freq).mean()
scaled_power_df = power_df.copy()

for c in scaled_power_df.columns:
    scaled_power_df[c] = power_df[c].values/metadata_df.loc[c].capacity
    
# scale between [0,1]/ or divide by total capacity
# Select zone
plant_ids = list(metadata_df[metadata_df['load_zone']==target_zone].index)

print('Number of plants per zone')
print(metadata_df.groupby(['load_zone'])['config'].count())

fig, ax = plt.subplots(constrained_layout = True)
metadata_df.plot(kind='scatter', x = 'longitude', y = 'latitude', ax = ax)
plt.show()

#%%

print(f'Target plant:{target_park}')
# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include

max_lag = 3 + min_lag

# min_lag = config['min_lag']
# max_lag = config['max_lag']

power_df = power_df
Y, Predictors, pred_col = create_IDsupervised(target_park, scaled_power_df[plant_ids], min_lag, max_lag)
#%%
target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = config['start_date']
split = config['split_date']
end = config['end_date']

# Predictors with weather
# trainPred = Predictors[start:split].dropna()
# testPred = Predictors[split:end].dropna()

# Predictors with weather
trainPred = pd.merge(Predictors[start:split], id_forecasts_df[start:split], how='inner', left_index=True, right_index=True).dropna()
testPred = pd.merge(Predictors[split:end], id_forecasts_df[split:end], how='inner', left_index=True, right_index=True).dropna()

trainY = Y[trainPred.index[0]:trainPred.index[-1]].values
testY = Y[testPred.index[0]:testPred.index[-1]].values
Target = Y[testPred.index[0]:testPred.index[-1]]


#%%%% Tune the number of lags using a linear regression

# potential_lags = np.arange(1, 7)
# loss = []
# for lag in potential_lags:

#     Y, Predictors, pred_col = create_IDsupervised(target_park, scaled_power_df[plant_ids], min_lag, min_lag + lag)

#     start = config['start_date']
#     split = config['split_date']
#     end = config['end_date']

#     trainPred = Predictors[start:split].dropna()
#     testPred = Predictors[split:end].dropna()
    
#     trainY = Y[trainPred.index[0]:trainPred.index[-1]].values
#     testY = Y[testPred.index[0]:testPred.index[-1]].values
#     Target = Y[testPred.index[0]:testPred.index[-1]]
    
#     ### Linear models: linear regression, ridge, lasso 
#     lr = LinearRegression(fit_intercept = True)
#     lr.fit(trainPred, trainY)
#     lr_pred = lr.predict(testPred).reshape(-1,1)

#     loss.append(100*mae(lr_pred, Target.values))

# max_lag = potential_lags[np.argmin(loss)] + min_lag

#%%%% Base Performance: Evaluate Forecasts without Missing Values

base_Predictions = pd.DataFrame(data = [])

### Linear models: linear regression, ridge, lasso 

# Hyperparameter tuning with by cross-validation
param_grid = {"alpha": [10**pow for pow in range(-5,2)]}

ridge = GridSearchCV(Ridge(fit_intercept = True, max_iter = 10_000), param_grid)
ridge.fit(trainPred, trainY)

lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 10_000), param_grid)
lasso.fit(trainPred, trainY)

lr = LinearRegression(fit_intercept = True)
lr.fit(trainPred, trainY)

alpha_best = lasso.best_params_['alpha']

lad = QR_regressor(fit_intercept = True)
lad.fit(trainPred, trainY)

lad_l1 = QR_regressor(fit_intercept = True, alpha = alpha_best)
lad_l1.fit(trainPred, trainY)

lr_pred= projection(lr.predict(testPred).reshape(-1,1))
lasso_pred = projection(lasso.predict(testPred).reshape(-1,1))
ridge_pred = projection(ridge.predict(testPred).reshape(-1,1))
lad_pred = projection(lad.predict(testPred).reshape(-1,1))
lad_l1_pred = projection(lad_l1.predict(testPred).reshape(-1,1))

persistence_pred = Target.values[:-config['min_lag']]
for i in range(config['min_lag']):
    persistence_pred = np.insert(persistence_pred, 0, trainY[-(1+i)]).reshape(-1,1)

base_Predictions['Persistence'] = persistence_pred.reshape(-1)
base_Predictions['LS'] = lr_pred.reshape(-1)
base_Predictions['Lasso'] = lasso_pred.reshape(-1)
base_Predictions['Ridge'] = ridge_pred.reshape(-1)
base_Predictions['LAD'] = lad_pred.reshape(-1)
base_Predictions['LAD-L1'] = lad_l1_pred.reshape(-1)
base_Predictions['Climatology'] = trainY.mean()
# base_Predictions['NREL'] = testPred[f'{target_park}_ID_for'].values

#%% Neural Network: train a standard MLP model


batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15
val_perc = 0.15

torch.manual_seed(0)

# Standard MLPs (separate) forecasting wind production and dispatch decisions
n_valid_obs = int(val_perc*len(trainY))

tensor_trainY = torch.FloatTensor(trainY[:-n_valid_obs])
tensor_validY = torch.FloatTensor(trainY[-n_valid_obs:])
tensor_testY = torch.FloatTensor(testY)

tensor_trainPred = torch.FloatTensor(trainPred.values[:-n_valid_obs])
tensor_validPred = torch.FloatTensor(trainPred.values[-n_valid_obs:])
tensor_testPred = torch.FloatTensor(testPred.values)

#### NN model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

mlp_model = MLP(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, 
                projection = True)

optimizer = torch.optim.Adam(mlp_model.parameters(), lr = learning_rate, weight_decay = 1e-5)
mlp_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                      patience = patience, verbose = 1)
base_Predictions['NN'] = projection(mlp_model.predict(testPred.values))
#%%
# Estimate MAE for base models
print('Base Model Performance, no missing data')

base_mae = pd.DataFrame(data = [], columns = base_Predictions.columns)
base_rmse = pd.DataFrame(data = [], columns = base_Predictions.columns)

for c in base_mae.columns: 
    base_mae[c] = [mae(base_Predictions[c].values, Target.values)]
    base_rmse[c] = [rmse(base_Predictions[c].values, Target.values)]

print((100*base_mae.mean()).round(2))
print((100*base_rmse.mean()).round(2))

# check forecasts visually
plt.plot(Target[:60].values)
plt.plot(lr_pred[:60])
plt.plot(lad_pred[:60])
plt.plot(lasso_pred[:60])
plt.plot(persistence_pred[:60])
plt.show() 

#%%
if config['save']:
    with open(f'{trained_models_path}\\{target_park}_LR_weather.pickle', 'wb') as handle:
        pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{trained_models_path}\\{target_park}_LAD_weather.pickle', 'wb') as handle:
        pickle.dump(lad, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{trained_models_path}\\{target_park}_Ridge_weather.pickle', 'wb') as handle:
        pickle.dump(ridge, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{trained_models_path}\\{target_park}_Lasso_weather.pickle', 'wb') as handle:
        pickle.dump(lasso, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{trained_models_path}\\{target_park}_MLP_weather.pickle', 'wb') as handle:
        pickle.dump(mlp_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%%%%%%%% Adversarial Models

target_pred = Predictors.columns
fixed_pred = [f'{target_park}_ID_for']
target_col = [np.where(trainPred.columns == c)[0][0] for c in target_pred]
fix_col = [np.where(trainPred.columns == c)[0][0] for c in fixed_pred]
        
###### Finitely Adaptive - LEARN partitions - LDR


n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15
val_perc = 0.15
decay = 1e-5
apply_LDR = True
LR_hidden_size = []
NN_hidden_size = [50, 50, 50]

###### LR base model 
try:
    with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_LDR_LR_models_dict = pickle.load(handle)
except:
    FA_LEARN_LDR_LR_models_dict = {}

Max_number_splits = [1, 2, 5, 20]

config['train'] = True
config['save'] = True

if config['train']:    
    for number_splits in Max_number_splits:
        FA_LEARN_LDR_LR_model = Learn_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, Max_models = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                                                    budget_constraint = 'inequality', attack_type = 'greedy', apply_LDR = apply_LDR)
                
        FA_LEARN_LDR_LR_model.fit(trainPred.values, trainY, val_split = val_perc, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                             lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False)
            
        FA_LEARN_LDR_LR_models_dict[number_splits] = FA_LEARN_LDR_LR_model
    
        if config['save']:
            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_LR_model_{number_splits}_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LDR_LR_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LDR_LR_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
###### NN base model 
try:
    with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_LDR_NN_models_dict = pickle.load(handle)
except:
    FA_LEARN_LDR_NN_models_dict = {}

Max_number_splits = [1, 2, 5, 20]


config['train'] = True
config['save'] = True

if config['train']:    
    for number_splits in Max_number_splits:
        FA_LEARN_LDR_NN_model = Learn_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, Max_models = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [50,50,50], output_size = n_outputs, 
                                                    budget_constraint = 'inequality', attack_type = 'greedy', apply_LDR = apply_LDR)
                
        FA_LEARN_LDR_NN_model.fit(trainPred.values, trainY, val_split = val_perc, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                             lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False, 
                             warm_start_nominal = False)
            
        FA_LEARN_LDR_NN_models_dict[number_splits] = FA_LEARN_LDR_NN_model
    
        if config['save']:
            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_NN_model_{number_splits}_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LDR_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LDR_NN_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%

###### Finitely Adaptive - FIXED partitions - LDR

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15
val_perc = 0.15
decay = 1e-5
apply_LDR = True
LR_hidden_size = []
NN_hidden_size = [50, 50, 50]

###### LR base model 
config['train'] = False
config['save'] = True

if config['train']:    
    FA_FIXED_LDR_LR_model = Fixed_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, 
                                                input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                                                budget_constraint = 'inequality', attack_type = 'random_sample', apply_LDR = apply_LDR)
            
    FA_FIXED_LDR_LR_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay)
        
    
    if config['save']:
        with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LDR_LR_model_weather.pickle', 'wb') as handle:
            pickle.dump(FA_FIXED_LDR_LR_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

###### NN base model 
config['train'] = False
config['save'] = True

if config['train']:    
    FA_FIXED_LDR_NN_model = Fixed_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, 
                                                         input_size = n_features, hidden_sizes = [50,50,50], output_size = n_outputs, 
                                                         budget_constraint = 'inequality', attack_type = 'random_sample', apply_LDR = apply_LDR)
            
    FA_FIXED_LDR_NN_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False)
        
    if config['save']:
        with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LDR_NN_model_weather.pickle', 'wb') as handle:
            pickle.dump(FA_FIXED_LDR_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)                
            
#%%
###### Finitely Adaptive - LEARN partitions - Static robust (**no LDR**)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15
val_perc = 0.15
decay = 1e-5
apply_LDR = False
LR_hidden_size = []
NN_hidden_size = [50, 50, 50]

###### LR base model 
try:
    with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_LR_models_dict = pickle.load(handle)
except:
    FA_LEARN_LR_models_dict = {}

Max_number_splits = [1, 2, 5, 20]

config['train'] = True
config['save'] = True

if config['train']:    
    for number_splits in Max_number_splits:
        FA_LEARN_LR_model = Learn_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, Max_models = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                                                    budget_constraint = 'inequality', attack_type = 'greedy', apply_LDR = apply_LDR)
                
        FA_LEARN_LR_model.fit(trainPred.values, trainY, val_split = val_perc, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                             lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False)
            
        FA_LEARN_LR_models_dict[number_splits] = FA_LEARN_LR_model
    
        if config['save']:
            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LR_model_{number_splits}_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LR_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_LR_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

###### NN base model 
try:
    with open(f'{trained_models_path}\\{target_park}_FA_LEARN_NN_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_NN_models_dict = pickle.load(handle)
except:
    FA_LEARN_NN_models_dict = {}

Max_number_splits = [1, 2, 5, 20]

config['train'] = True
config['save'] = True

if config['train']:    
    for number_splits in Max_number_splits:
        FA_LEARN_NN_model = Learn_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, Max_models = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [50,50,50], output_size = n_outputs, 
                                                    budget_constraint = 'inequality', attack_type = 'greedy', apply_LDR = apply_LDR)
                
        FA_LEARN_NN_model.fit(trainPred.values, trainY, val_split = val_perc, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                             lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False)
            
        FA_LEARN_NN_models_dict[number_splits] = FA_LEARN_NN_model
    
        if config['save']:
            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_NN_model_{number_splits}_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{trained_models_path}\\{target_park}_FA_LEARN_NN_models_dict_weather.pickle', 'wb') as handle:
                pickle.dump(FA_LEARN_NN_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
###### Finitely Adaptive - FIXED partitions - Static Robust (**no LDR**)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15
val_perc = 0.15
decay = 1e-5
apply_LDR = False
LR_hidden_size = []
NN_hidden_size = [50, 50, 50]

###### LR base model 
config['train'] = False
config['save'] = True

if config['train']:    
    FA_FIXED_LR_model = Fixed_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, 
                                                input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                                                budget_constraint = 'inequality', attack_type = 'random_sample', apply_LDR = apply_LDR)
            
    FA_FIXED_LR_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay)
        
    
    if config['save']:
        with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LR_model_weather.pickle', 'wb') as handle:
            pickle.dump(FA_FIXED_LR_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

###### NN base model 
config['train'] = False
config['save'] = True

if config['train']:    
    FA_FIXED_NN_model = Fixed_FiniteAdapt_Robust_Reg(target_col = target_col, fix_col = fix_col, 
                                                         input_size = n_features, hidden_sizes = [50,50,50], output_size = n_outputs, 
                                                         budget_constraint = 'inequality', attack_type = 'random_sample', apply_LDR = apply_LDR)
            
    FA_FIXED_NN_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay, freeze_weights = False)
        
    if config['save']:
        with open(f'{trained_models_path}\\{target_park}_FA_FIXED_NN_model_weather.pickle', 'wb') as handle:
            pickle.dump(FA_FIXED_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)                