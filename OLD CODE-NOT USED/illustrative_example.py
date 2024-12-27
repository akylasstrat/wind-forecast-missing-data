# -*- coding: utf-8 -*-
"""
Illustrative example

@author: a.stratigakos
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from utility_functions import * 

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

    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 2 # per pair of (n_nodes,percentage)
    params['pattern'] = 'MCAR'
    params['retrain'] = False
    
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)
    params['train'] = True # If True, then train models, else tries to load previous runs
    params['save'] = True # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()


#%%
np.random.seed(0)

# generate feature data (see Appendix E.1. in Beyond Impute-then-Regresss)
num_feat = 3
n = 10000
# Correlation matrix
b = np.random.uniform(0, 0.25, size=(3,3))
S = (b + b.T)/2

X = np.random.normal(size = (n, num_feat))

w = np.random.uniform(size = (num_feat))
bias = np.random.normal(1)

Y = X@w + bias

trainY = Y[:1000].reshape(-1,1)
trainX = X[:1000]

#%%%%%%%%% Adversarial Models

target_col = [0, 1, 2]
fix_col = []

#%%###### Finitely Adaptive - greedy partitions - LS model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle


batch_size = 256
num_epochs = 250
learning_rate = 1e-3
patience = 50

FA_greedy_LS_model = FiniteAdapt_Greedy(target_col = target_col, fix_col = fix_col, Max_splits = 10, D = 1_000, red_threshold = 1e-10, 
                                            input_size = num_feat, hidden_sizes = [], output_size = 1, projection = False, 
                                            train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')

FA_greedy_LS_model.fit(trainX, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 1e-10, epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                      lr = learning_rate, batch_size = batch_size, weight_decay = 0)

#%%
# find tree leaves
leaf_ind = np.where(np.array(FA_greedy_LS_model.feature)==-1)[0]

#%%
print('Missing Patterns of Leaf Indices')
print(np.array(FA_greedy_LS_model.missing_pattern)[leaf_ind])

#%%
plt.plot(np.array(FA_greedy_LS_model.Loss))
plt.plot(np.array(FA_greedy_LS_model.WC_Loss))

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
    
FA_lin_greedy_LS_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_splits = number_splits, D = 1_000, red_threshold = 1e-5, 
                                            input_size = n_features, hidden_sizes = [], output_size = n_outputs, projection = True, 
                                            train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')

FA_lin_greedy_LS_model.fit(trainPred.values, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                      epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                      lr = 1e-3, batch_size = batch_size, weight_decay = 0)
    

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
