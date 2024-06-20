# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:59:37 2024

@author: astratig
"""

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

power_df = pd.read_csv('C:\\Users\\astratig\\feature-deletion-robust\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0)
metadata_df = pd.read_csv('C:\\Users\\astratig\\feature-deletion-robust\\data\\smart4res_data\\wind_metadata.csv', index_col=0)

# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)
# transition matrix to generate missing data/ estimated from training data (empirical estimation)
P = np.array([[0.999, 0.001], [0.241, 0.759]])

plt.figure(constrained_layout = True)
plt.scatter(x=metadata_df['Long'], y=metadata_df['Lat'])
plt.show()

#%%
target_park = 'p_1088'

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

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_MLP.pickle', 'rb') as handle:
    mlp_model = pickle.load(handle)

#%%%%%%%%% Adversarial Models

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []
        
#%%
###### Finitely *Linearly* Adaptive - greedy partitions - LS model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

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

# Finitely Adaptive - Linear - Fixed partitions

from torch_custom_layers import * 
from finite_adaptability_model_functions import *

########## LS model
batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15

config['train'] = False
config['save'] = False

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

config['train'] = False
config['save'] = False

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
#%%

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


from finite_adaptability_model_functions import *
from torch_custom_layers import *

for gamma  in np.arange(10, 15):
    
    print(f'Budget:{gamma}')
    
    # Train robust model
    print('Output correction')
    lin_model = adjustable_FDR(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = 1, 
                              target_col = target_col, fix_col = fix_col, projection = False, 
                              Gamma = gamma, train_adversarially = True, budget_constraint = 'equality')
    
    optimizer = torch.optim.Adam(lin_model.parameters(), lr = 1e-3, weight_decay = 1e-5)

    lin_model.load_state_dict(mlp_model.state_dict(), strict = False)      
    
    # Warm-start: use nominal model or previous iteration
    # lin_model.model[0].weight.data = torch.FloatTensor(lr.coef_[0].reshape(1,-1))
    # lin_model.model[0].bias.data = torch.FloatTensor(lr.intercept_)
                
    lin_model.adversarial_train_model(train_base_data_loader, valid_base_data_loader, optimizer, 
                                epochs = 250, patience = 15, verbose = 0, attack_type = 'random_sample', 
                                freeze_weights = False)

    print(f'Best val loss:{lin_model.best_val_loss}')

    # Train robust model
    
    print('Input correction')
    v2_lin_model = adjustable_FDR(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = 1, 
                              target_col = target_col, fix_col = fix_col, projection = False, 
                              Gamma = gamma, train_adversarially = True, budget_constraint = 'equality')
    
    optimizer = torch.optim.Adam(v2_lin_model.parameters(), lr = 1e-3, weight_decay = 0)

    # Warm-start using nominal model    
    v2_lin_model.load_state_dict(mlp_model.state_dict(), strict = False)      

    # for j in range(0, 3*2+2, 2):
    #     v2_lin_model.model[j].weight.data = mlp_model.model[j].weight.data.clone()
    #     v2_lin_model.model[j].bias.data = mlp_model.model[j].bias.data.clone()
    
    # Warm-start: use nominal model or previous iteration
    # v2_lin_model.linear_correction_layer.weight.data = torch.FloatTensor(lr.coef_[0].reshape(1,-1))
    # v2_lin_model.linear_correction_layer.bias.data = torch.FloatTensor(lr.intercept_)
                
    v2_lin_model.adversarial_train_model(train_base_data_loader, valid_base_data_loader, optimizer, 
                               epochs = 250, patience = 15, verbose = 0, attack_type = 'random_sample', 
                               freeze_weights = False)
    
    print(f'Best val loss:{v2_lin_model.best_val_loss}')

#%%
from finite_adaptability_model_functions import *
from torch_custom_layers import *

FA_lin_fixed_LS_model = FiniteAdapt_Linear_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [], 
                                output_size = n_outputs, projection = True, train_adversarially = True)

FA_lin_fixed_LS_model.fit(trainPred.values, trainY, val_split = 0, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                     lr = learning_rate, batch_size = batch_size, weight_decay = 0)
#%%

from finite_adaptability_model_functions import *
from torch_custom_layers import *


v2_FA_lin_fixed_LS_model = v2_FiniteAdapt_Linear_Fixed(target_col = target_col, fix_col = fix_col, input_size = n_features, hidden_sizes = [], 
                                output_size = n_outputs, projection = True, train_adversarially = True)

v2_FA_lin_fixed_LS_model.fit(trainPred.values, trainY, val_split = 0, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                     lr = learning_rate, batch_size = batch_size, weight_decay = 0)









