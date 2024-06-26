# -*- coding: utf-8 -*-
"""
Main model training 

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
from FDR_regressor import *
from QR_regressor import *

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
    params['max_lag'] = 3
    params['min_lag'] = 1  #!!! do not change this for the moment

    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2019-01-01' # start of train set
    params['split_date'] = '2019-06-01' # end of train set/start of test set
    params['end_date'] = '2020-01-01'# end of test set
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 2 # per pair of (n_nodes,percentage)
    params['pattern'] = 'MCAR'
    params['retrain'] = False
    
    params['min_lag'] = 4
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)
    params['train'] = True # If True, then train models, else tries to load previous runs
    params['save'] = True # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0)
metadata_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_metadata.csv', index_col=0)

# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)
# transition matrix to generate missing data/ estimated from training data (empirical estimation)
P = np.array([[0.999, 0.001], [0.241, 0.759]])

plt.figure(constrained_layout = True)
plt.scatter(x=metadata_df['Long'], y=metadata_df['Lat'])
plt.show()

#%%
target_park = 'p_1475'

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
config['min_lag'] = 1
config['max_lag'] = 2 + config['min_lag']

min_lag = config['min_lag']
max_lag = config['max_lag']

power_df = power_df
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = config['start_date']
split = config['split_date']
end = config['end_date']

trainY = Y[start:split].values
testY = Y[split:end].values
Target = Y[split:end]

trainPred = Predictors[start:split]
testPred = Predictors[split:end]


#%%%% Tune the number of lags using a linear regression

# potential_lags = np.arange(1, 7)
# loss = []
# for lag in potential_lags:

#     Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, min_lag + lag)

#     target_scaler = MinMaxScaler()
#     pred_scaler = MinMaxScaler()

#     start = '2019-01-01'
#     split = '2019-04-01'
#     end = '2020-01-01'

#     trainY = target_scaler.fit_transform(Y[start:split].values)
#     testY = target_scaler.transform(Y[split:end])
#     Target = Y[split:end]
    
#     trainPred = pred_scaler.fit_transform(Predictors[start:split])
#     testPred = pred_scaler.transform(Predictors[split:end])

#     ### Linear models: linear regression, ridge, lasso 
#     lr = LinearRegression(fit_intercept = True)
#     lr.fit(trainPred, trainY)
#     lr_pred = target_scaler.inverse_transform(lr.predict(testPred).reshape(-1,1))

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

#%% Neural Network: train a standard MLP model

from torch_custom_layers import * 

batch_size = 512
num_epochs = 1000
learning_rate = 1e-3
patience = 25

torch.manual_seed(0)

# Standard MLPs (separate) forecasting wind production and dispatch decisions
n_valid_obs = int(0.15*len(trainY))

tensor_trainY = torch.FloatTensor(trainY[:-n_valid_obs])
tensor_validY = torch.FloatTensor(trainY[-n_valid_obs:])
tensor_testY = torch.FloatTensor(testY)

tensor_trainPred = torch.FloatTensor(trainPred.values[:-n_valid_obs])
tensor_validPred = torch.FloatTensor(trainPred.values[-n_valid_obs:])
tensor_testPred = torch.FloatTensor(testPred.values)

#### MLP model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

mlp_model = MLP(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, 
                projection = True)

optimizer = torch.optim.Adam(mlp_model.parameters(), lr = learning_rate, weight_decay = 1e-5)
mlp_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                      patience = patience, verbose = 0)
base_Predictions['NN'] = projection(mlp_model.predict(testPred.values))

# Estimate MAE for base models
print('Base Model Performance, no missing data')
from utility_functions import *

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
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_LR.pickle', 'wb') as handle:
        pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_LAD.pickle', 'wb') as handle:
        pickle.dump(lad, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_Ridge.pickle', 'wb') as handle:
        pickle.dump(ridge, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_Lasso.pickle', 'wb') as handle:
        pickle.dump(lasso, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_MLP.pickle', 'wb') as handle:
        pickle.dump(mlp_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%%%%%%%% Adversarial Models

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []


###### Finitely Adaptive - fixed partitions - LAD model
from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
import pickle

# config['train'] = False
# config['save'] = False

config['train'] = True

if config['train']:
    FA_greedy_LAD_model = depth_Finite_FDRR(Max_models = 50, D = 1_000, red_threshold = 1e-5, max_gap = 0.05)
    FA_greedy_LAD_model.fit(trainPred.values, trainY, target_col, fix_col, tree_grow_algo = 'leaf-wise', 
                      budget = 'inequality', solution = 'reformulation')
    
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LAD_model.pickle', 'wb') as handle:
            pickle.dump(FA_greedy_LAD_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    try:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LAD_model.pickle', 'rb') as handle:    
                FA_greedy_LAD_model = pickle.load(handle)

            
    except:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LAD_model.pickle', 'rb') as handle:    
                temp_model = pickle.load(handle)
                
        FA_greedy_LAD_model = temp_model

        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LAD_model.pickle', 'wb') as handle:
            pickle.dump(FA_greedy_LAD_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#%%
###### Finitely *Linearly* Adaptive - greedy partitions - LS model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 1000
learning_rate = 1e-2
patience = 15

Max_number_splits = [1, 2, 5, 10, 25]
FA_lin_greedy_LS_models_dict = {}

config['train'] = False

if config['train']:
    
    for number_splits in Max_number_splits:
        FA_lin_greedy_LS_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_splits = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [], output_size = n_outputs, projection = True, 
                                                    train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
        
        FA_lin_greedy_LS_model.fit(trainPred.values, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                              epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                              lr = 1e-3, batch_size = batch_size, weight_decay = 0)
    
        FA_lin_greedy_LS_models_dict[number_splits] = FA_lin_greedy_LS_model
    
        if config['save']:
            with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_model_{number_splits}.pickle', 'wb') as handle:
                pickle.dump(FA_lin_greedy_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'wb') as handle:
                pickle.dump(FA_lin_greedy_LS_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    

else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'rb') as handle:
            FA_lin_greedy_LS_models_dict = pickle.load(handle)
            
#%%
# t = -4
# print(fin_LS_model.missing_pattern[t])
# print(fin_LS_model.fixed_features[t])
# print('Coef')
# print(fin_LS_model.node_model_[t].coef_[0].round(2))

# for layer in fin_LS_models_dict[5].wc_node_model_[t].model.children():        
#     if isinstance(layer, nn.Linear):    
#         plt.plot(layer.weight.data.detach().numpy().T, label = 'wc coeff')
#         w = layer.weight.data.detach().numpy()
#         print('WC Coeff')
#         print(layer.weight.data.detach().numpy().round(2))

#     plt.plot(fin_LS_models_dict[5].node_model_[t].coef_[0],'--' , label = 'nominal case')
#     # plt.plot(lr.coef_.T)
#     plt.legend()
#     plt.show()

# W = fin_LS_model.wc_node_model_[t].W.detach().numpy().T

#%%###### Finitely Adaptive - greedy partitions - LS model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25

config['train'] = True
config['save'] = True

if config['train']:
    FA_greedy_LS_model = FiniteAdapt_Greedy(target_col = target_col, fix_col = fix_col, Max_splits = 10, D = 1_000, red_threshold = 1e-5, 
                                                input_size = n_features, hidden_sizes = [], output_size = n_outputs, projection = True, 
                                                train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
    
    FA_greedy_LS_model.fit(trainPred.values, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                          epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                          lr = learning_rate, batch_size = batch_size, weight_decay = 0)
    
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LS_model.pickle', 'wb') as handle:
            pickle.dump(FA_greedy_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
else:

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LS_model.pickle', 'rb') as handle:
            FA_greedy_LS_model = pickle.load(handle)

#%%
###### Finitely Adaptive - Greedy partitions - NN model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25

config['train'] = True
config['save'] = True

if config['train']:
    
    FA_greedy_NN_model = FiniteAdapt_Greedy(target_col = target_col, fix_col = fix_col, Max_splits = 10, D = 1_000, red_threshold = 1e-5, 
                                                input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, projection = True, 
                                                train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
    
    FA_greedy_NN_model.fit(trainPred.values, trainY, val_split = 0.15, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                          epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                          lr = learning_rate, batch_size = batch_size, weight_decay = 1e-5)
    
    
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_NN_model.pickle', 'wb') as handle:
            pickle.dump(FA_greedy_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
else:

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_NN_model.pickle', 'rb') as handle:
            FA_greedy_NN_model = pickle.load(handle)


#%%
###### Finitely *Linearly* Adaptive - learning partitions - NN model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

# Standard MLPs (separate) forecasting wind production and dispatch decisions
n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

#optimizer = torch.optim.Adam(res_mlp_model.parameters())
target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25

config['train'] = True
config['save'] = True

if config['train']:
            
    FA_lin_greedy_NN_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_models = 10, D = 1_000, red_threshold = 1e-5, 
                                                input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, projection = True, 
                                                train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
    
    FA_lin_greedy_NN_model.fit(trainPred.values, trainY, val_split = 0.15, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                          epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = 1e-5)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_NN_model.pickle', 'wb') as handle:
            pickle.dump(FA_lin_greedy_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    try:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_NN_model.pickle', 'rb') as handle:    
                FA_lin_greedy_NN_model = pickle.load(handle)
    except:
        
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_NN_model.pickle', 'rb') as handle:    
                temp_model = pickle.load(handle)
        FA_lin_greedy_NN_model = temp_model

        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_NN_model.pickle', 'wb') as handle:
            pickle.dump(FA_lin_greedy_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% 
###### Finetely Adaptive - Fixed partitions -  LS & NN model (approximates FDRR from previous work)
# Train one model per each integer value in range [0, gamma]
from torch_custom_layers import * 
from finite_adaptability_model_functions import *

# case_folder = config['store_folder']
# output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_fdrr-aar_{max_lag}.pickle'

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

#!!!!! Fix validation set

# Standard MLPs (separate) forecasting wind production and dispatch decisions
tensor_trainY = torch.FloatTensor(trainY)
tensor_validY = torch.FloatTensor(trainY)
tensor_testY = torch.FloatTensor(testY)

tensor_trainPred = torch.FloatTensor(trainPred.values)
tensor_validPred = torch.FloatTensor(trainPred.values)
tensor_testPred = torch.FloatTensor(testPred.values)

### MLP model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

########## LS model

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15

config['train'] = True
config['save'] = True

if config['train']:
            
    FA_fixed_LS_model = FiniteAdapt_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [], 
                                    output_size = n_outputs, projection = True, train_adversarially = True)
    
    FA_fixed_LS_model.fit(trainPred.values, trainY, val_split = 0, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = 0)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_LS_model.pickle', 'wb') as handle:
            pickle.dump(FA_fixed_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_LS_model.pickle', 'rb') as handle:    
            FA_fixed_LS_model = pickle.load(handle)

for j in [0, 1, 2, 3, 4]:
    for layer in FA_fixed_LS_model.FDR_models[j].model.children():
        if isinstance(layer, nn.Linear):    
            plt.plot(layer.weight.data.detach().numpy().T, label = 'GD')
# plt.legend()
plt.show()

########## NN model

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25
val_perc = 0.15
decay = 1e-5

config['train'] = True
config['save'] = True

if config['train']:
            
    FA_fixed_NN_model = FiniteAdapt_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [50, 50, 50], 
                                    output_size = n_outputs, projection = True, train_adversarially = True)
    
    FA_fixed_NN_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_NN_model.pickle', 'wb') as handle:
            pickle.dump(FA_fixed_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_NN_model.pickle', 'rb') as handle:    
            FA_fixed_NN_model = pickle.load(handle)

# Finitely Adaptive - Linear - Fixed partitions

from torch_custom_layers import * 
from finite_adaptability_model_functions import *

########## LS model
batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15

config['train'] = True
config['save'] = True

if config['train']:
            
    FA_lin_fixed_LS_model = FiniteAdapt_Linear_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [], 
                                    output_size = n_outputs, projection = True, train_adversarially = True)
    
    FA_lin_fixed_LS_model.fit(trainPred.values, trainY, val_split = 0, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = 0)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_LS_model.pickle', 'wb') as handle:
            pickle.dump(FA_lin_fixed_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_LS_model.pickle', 'rb') as handle:    
            FA_lin_fixed_LS_model = pickle.load(handle)

########## NN model

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25
val_perc = 0.15
decay = 1e-5

config['train'] = True
config['save'] = True

if config['train']:
            
    FA_lin_fixed_NN_model = FiniteAdapt_Linear_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [50, 50, 50], 
                                    output_size = n_outputs, projection = True, train_adversarially = True)
    
    FA_lin_fixed_NN_model.fit(trainPred.values, trainY, val_split = val_perc, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = decay)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_NN_model.pickle', 'wb') as handle:
            pickle.dump(FA_lin_fixed_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_NN_model.pickle', 'rb') as handle:    
            FA_lin_fixed_NN_model = pickle.load(handle)
