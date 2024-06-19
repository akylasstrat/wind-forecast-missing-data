# -*- coding: utf-8 -*-
"""
Finite adaptability

@author: a.stratigakos
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

def eval_dual_predictions(pred, target, priceup, pricedown):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''

    error = target.reshape(-1)-pred.reshape(-1)
    total_cost = (-priceup*error[error<0]).sum() + (pricedown*error[error>0]).sum()

    return (1/len(target))*total_cost

def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

def retrain_model(X, Y, testX, target_col, fix_col, Gamma, base_loss = 'l2'):
    ''' Retrain model without missing features
        returns a list models and corresponding list of missing features'''
    # all combinations of missing features for up to Gamma features missing
    combinations = [list(item) for sublist in [list(itertools.combinations(range(len(target_col)), gamma)) for gamma in range(1,Gamma+1)] for item in sublist]
    # first instance is without missing features
    print(f'Number of models: {len(combinations)}')
    combinations.insert(0, [])
    models = []
    predictions = []
    
    # augment feature matrix with dummy variable to avoid infeasible solution when all features are missing
    # !!!!! set fit_intercept = False in lr model.
    
    fix_col_bias = fix_col + [X.shape[1]]
    augm_X = np.column_stack((X.copy(), np.ones(len(X))))
    augm_testX = np.column_stack((testX.copy(), np.ones(len(testX))))

    for i,v in enumerate(combinations):
        
        # find columns not missing 
        #temp_col = [col for col in target_col if col not in v]
        #temp_X = X[:,temp_col+fix_col]
        #temp_test_X = testX[:,temp_col+fix_col]
        
        # find columns not missing 
        temp_col = [col for col in target_col if col not in v]
        temp_X = augm_X[:,temp_col+fix_col_bias]
        temp_test_X = augm_testX[:,temp_col+fix_col_bias]

        # retrain model without missing features
        if base_loss == 'l2':
            lr = LinearRegression(fit_intercept = True)
            lr.fit(temp_X, Y)
        elif base_loss == 'l1':
            lr = QR_regressor()
            lr.fit(temp_X, Y)
            
        models.append(lr)
        predictions.append(lr.predict(temp_test_X).reshape(-1))
    
    predictions = np.array(predictions).T
    
    return models, predictions, combinations

def create_IDsupervised(target_col, df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    lead_time_name = '-' + target_col + '_t'+str(min_lag)
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for park in p_df.columns:
        for lag in range(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    Y = p_df[target_col].to_frame()
    
    return Y, Predictors, pred_col

def create_feat_matrix(df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for park in p_df.columns:
        for lag in np.arange(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    return Predictors

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['scale'] = False
    params['K value'] = 5 #Define budget of uncertainty value
    params['impute'] = True # If True, apply mean imputation for missing features
    params['cap'] = False # If True, apply dual constraints for capacity (NOT YET IMPLEMENTED)
    params['trainReg'] = False #Determine best value of regularization for given values. 
 
    params['store_folder'] = 'ID-case' # folder to save stuff (do not change)
    params['max_lag'] = 3
    params['min_lag'] = 1  #!!! do not change this for the moment

    # Penalties for imbalance cost (only for fixed prices)
    params['pen_up'] = 4 
    params['pen_down'] = 3 

    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2019-01-01' # start of train set
    params['split_date'] = '2020-01-01' # end of train set/start of test set
    params['end_date'] = '2020-05-01'# end of test set
    
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
target_park = 'p_1088'

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
# config['min_lag'] = 4
config['max_lag'] = 2 + config['min_lag']

min_lag = config['min_lag']
max_lag = config['max_lag']

power_df = power_df
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = '2019-01-01'
split = '2019-06-01'
end = '2020-01-01'

if config['scale']:
    trainY = target_scaler.fit_transform(Y[start:split].values)
    testY = target_scaler.transform(Y[split:end])
    Target = Y[split:end]
    
    trainPred = pred_scaler.fit_transform(Predictors[start:split])
    testPred = pred_scaler.transform(Predictors[split:end])
else:
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

#     loss.append(mae(lr_pred, Target.values))

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

if config['scale']:
    lr_pred = target_scaler.inverse_transform(lr.predict(testPred).reshape(-1,1))
    lasso_pred = target_scaler.inverse_transform(lasso.predict(testPred).reshape(-1,1))
    ridge_pred = target_scaler.inverse_transform(ridge.predict(testPred).reshape(-1,1))
else:
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


### Finite Adaptability - Fixed - LAD
from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
import pickle

# config['train'] = False
# config['save'] = False

config['train'] = False

if config['train']:
    fin_LAD_model = depth_Finite_FDRR(Max_models = 50, D = 1_000, red_threshold = 1e-5, max_gap = 0.05)
    fin_LAD_model.fit(trainPred.values, trainY, target_col, fix_col, tree_grow_algo = 'leaf-wise', 
                      budget = 'inequality', solution = 'reformulation')
    
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LAD_model.pickle', 'wb') as handle:
            pickle.dump(fin_LAD_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LAD_model.pickle', 'rb') as handle:    
            fin_LAD_model = pickle.load(handle)

# plt.plot(np.array(fin_LAD_model.Loss))
# plt.plot(np.array(fin_LAD_model.ineq_wc_Loss))
# plt.plot(np.array(fin_LAD_model.eq_wc_Loss))
# plt.show()
#%% Finite Adaptability - Linear - greedy partitions - LS base model

from FiniteRetrain import *
from FiniteRobustRetrain import *
from finite_adaptability_model_functions import *
from torch_custom_layers import *
import pickle

# Standard MLPs (separate) forecasting wind production and dispatch decisions
n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

#optimizer = torch.optim.Adam(res_mlp_model.parameters())

batch_size = 512
num_epochs = 1000
learning_rate = 1e-2
patience = 15

# config['train'] = True
# config['save'] = True

Max_number_splits = [1, 2, 5, 10, 25]
fin_LS_models_dict = {}

config['train'] = False

if config['train']:
    
    for number_splits in Max_number_splits:
        fin_LS_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_splits = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [], output_size = n_outputs, projection = True, 
                                                    train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
        
        fin_LS_model.fit(trainPred.values, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                              epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                              lr = 1e-3, batch_size = batch_size, weight_decay = 0)
    
        fin_LS_models_dict[number_splits] = fin_LS_model
    
        if config['save']:
            with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LS_model_{number_splits}.pickle', 'wb') as handle:
                pickle.dump(fin_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LS_models_dict.pickle', 'wb') as handle:
                pickle.dump(fin_LS_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
else:
    # with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LS_model.pickle', 'rb') as handle:    
    #         fin_LS_model = pickle.load(handle)

    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_LS_models_dict.pickle', 'rb') as handle:
            fin_LS_models_dict = pickle.load(handle)

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

#%% Finitely Adaptive - Linear - NN

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
num_epochs = 1000
learning_rate = 1e-3
patience = 25

config['train'] = True
config['save'] = True

if config['train']:
            
    fin_NN_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_models = 10, D = 1_000, red_threshold = 1e-5, 
                                                input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, projection = True, 
                                                train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
    
    fin_NN_model.fit(trainPred.values, trainY, val_split = 0.15, tree_grow_algo = 'leaf-wise', max_gap = 1e-3, 
                          epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                         lr = learning_rate, batch_size = batch_size, weight_decay = 1e-5)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_NN_model.pickle', 'wb') as handle:
            pickle.dump(fin_NN_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_fin_NN_model.pickle', 'rb') as handle:    
            fin_NN_model = pickle.load(handle)


#%% Finetely adaptive - Constant - fixed partitions -  LS model (approximates FDRR from previous work)
# Train one model per each integer value in range [0, gamma]
from torch_custom_layers import * 

case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_fdrr-aar_{max_lag}.pickle'

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

K_parameter = np.arange(0, len(target_pred)+1)

FDRR_AAR_models = []

batch_size = 512
num_epochs = 250
learning_rate = 1e-2
patience = 15

gd_FDRR_R_models = []

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

config['train'] = True
config['save'] = True

if config['train']:
    for K in K_parameter:
        print(f'Budget: {K}')
        
        # sample missing data to check how predictions are formed
        feat = np.random.choice(target_col, size = K, replace = False)
        a = np.zeros((1,trainPred.shape[1]))
        a[:,feat] = 1
        test_alpha = torch.FloatTensor(a)
        
        gd_fdr_model = gd_FDRR(input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                                  target_col = target_col, fix_col = fix_col, projection = False, 
                                  Gamma = K, train_adversarially = True, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(gd_fdr_model.parameters(), lr = 1e-3)
    
        # Warm-start: use nominal model or previous iteration
        if K == 0:
            gd_fdr_model.model[0].weight.data = torch.FloatTensor(lr.coef_[0].reshape(1,-1))
            gd_fdr_model.model[0].bias.data = torch.FloatTensor(lr.intercept_)
        else:
            gd_fdr_model.load_state_dict(gd_FDRR_R_models[-1].state_dict(), strict=False)
            
        gd_fdr_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                              patience = patience, verbose = 0, warm_start = False, attack_type = 'random_sample')
    
        gd_fdr_pred = gd_fdr_model.predict(testPred.values, project = True)
        gd_FDRR_R_models.append(gd_fdr_model)
        
        print('GD FDRR: ', eval_point_pred(gd_fdr_pred, Target.values, digits=4))
        
        # plt.plot(gd_fdr_model.predict(tensor_testPred*(1-test_alpha))[:1000], label='GD')
        # plt.plot(projection(gd_FDRR_R_models[K].predict(testPred.values*(1-a))[:1000]), label='Eq-MinMax')
        # plt.legend()
        # plt.show()

        for layer in gd_fdr_model.model.children():
            if isinstance(layer, nn.Linear):    
                plt.plot(layer.weight.data.detach().numpy().T, label = 'GD')
        plt.legend()
        plt.show()

    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_gd_FDRR_R_model.pickle', 'wb') as handle:
            pickle.dump(gd_FDRR_R_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:    
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_gd_FDRR_R_model.pickle', 'rb') as handle:    
            gd_FDRR_R_models = pickle.load(handle)


for j in [0, 1, 2, 3, 4]:
    for layer in gd_FDRR_R_models[j].model.children():
        if isinstance(layer, nn.Linear):    
            plt.plot(layer.weight.data.detach().numpy().T, label = 'GD')
# plt.legend()
plt.show()


#%% Finetely adaptive - Linear - fixed partitions -  LS model (extends FDRR from previous work)

from torch_custom_layers import * 

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 15

ladj_FDRR_R_models = []

#!!!!! Fix validation set

### MLP model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

config['train'] = True
config['save'] = True

if config['train']:
    for K in K_parameter:
        print(f'Budget: {K}')
        
        # sample missing data to check how predictions are formed
        feat = np.random.choice(target_col, size = K, replace = False)
        a = np.zeros((1,trainPred.shape[1]))
        a[:,feat] = 1
        test_alpha = torch.FloatTensor(a)
        
        adj_fdr_model = adjustable_FDR(input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                          target_col = target_col, fix_col = fix_col, projection = False, 
                          Gamma = K, train_adversarially = True, budget_constraint = 'equality')

        optimizer = torch.optim.Adam(adj_fdr_model.parameters(), lr = 1e-3)
        
        # Warm-start using nominal model
        # adj_fdr_model.linear_correction_layer.weight.data = torch.FloatTensor(lr.coef_[0].reshape(1,-1))
        # adj_fdr_model.linear_correction_layer.bias.data = torch.FloatTensor(lr.intercept_.reshape(1,-1))

        adj_fdr_model.model[0].weight.data = torch.FloatTensor(lr.coef_[0].reshape(1,-1))
        adj_fdr_model.model[0].bias.data = torch.FloatTensor(lr.intercept_.reshape(1,-1))
            
        adj_fdr_model.adversarial_train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                              patience = patience, freeze_weights = False, attack_type = 'random_sample')
                    
        ladj_FDRR_R_models.append(adj_fdr_model)

    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_ladj_FDRR_R_models.pickle', 'wb') as handle:
            pickle.dump(ladj_FDRR_R_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:    
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_ladj_FDRR_R_models.pickle', 'rb') as handle:    
            ladj_FDRR_R_models = pickle.load(handle)
            
#% Retrain without missing features (Tawn, Browell)

# if config['retrain']:
#     retrain_models, retrain_pred, retrain_comb = retrain_model(trainPred, trainY, testPred, 
#                                                            target_col, fix_col, max(K_parameter), base_loss='l2')

#%% FDR - gradient-based, **NN** model
# Finetely adaptive - Constant - fixed partitions -  NN model (extends previous FDR to non-linear models)
# Train one model per each integer value in range [0, gamma]
from torch_custom_layers import * 

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25


# Standard MLPs (separate) forecasting wind production and dispatch decisions
n_valid_obs = int(0.15*len(trainY))

tensor_trainY = torch.FloatTensor(trainY[:-n_valid_obs])
tensor_validY = torch.FloatTensor(trainY[-n_valid_obs:])
tensor_testY = torch.FloatTensor(testY)

tensor_trainPred = torch.FloatTensor(trainPred.values[:-n_valid_obs])
tensor_validPred = torch.FloatTensor(trainPred.values[-n_valid_obs:])
tensor_testPred = torch.FloatTensor(testPred.values)

### MLP model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

config['train'] = True
config['save'] = True
#%%
FDR_NN_models = []

if config['train']:
    for K in K_parameter:
        print(f'Budget: {K}')
        
        torch.manual_seed(0)        
        # sample missing data to check how predictions are formed
        feat = np.random.choice(target_col, size = K, replace = False)
        a = np.zeros((1,trainPred.shape[1]))
        a[:,feat] = 1
        test_alpha = torch.FloatTensor(a)

        
        fdr_nn_model = gd_FDRR(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, 
                                  target_col = target_col, fix_col = fix_col, projection = False, 
                                  Gamma = K, train_adversarially = True, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(fdr_nn_model.parameters(), lr = 1e-3, weight_decay = 1e-5)

        # Warm-start: use nominal model or previous iteration
        if K == 0:
            fdr_nn_model.load_state_dict(mlp_model.state_dict(), strict = False)
        else:
            fdr_nn_model.load_state_dict(FDR_NN_models[-1].state_dict(), strict=False)
        
            fdr_nn_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                                  patience = patience, verbose = 0, warm_start = False, attack_type = 'random_sample')
    
        fdr_nn_pred = fdr_nn_model.predict(testPred.values, project = True)
        FDR_NN_models.append(fdr_nn_model)
        
        print('GD FDRR: ', eval_point_pred(fdr_nn_pred, Target.values, digits=4))
        

    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FDR_NN_models.pickle', 'wb') as handle:
            pickle.dump(FDR_NN_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:    
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FDR_NN_models.pickle', 'rb') as handle:    
            FDR_NN_models = pickle.load(handle)


#%% Finetely adaptive - Linearly Adaptive - fixed partitions -  **NN** model
# Train one model for each integer in range [0, gamma]

from torch_custom_layers import * 

batch_size = 512
num_epochs = 250
learning_rate = 1e-3
patience = 25

FDR_Lin_NN_models = []

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

config['train'] = True
config['save'] = True

if config['train']:
    for K in K_parameter:
        print(f'Budget: {K}')
        # sample missing data to check how predictions are formed   
        torch.manual_seed(0)

        fdr_lin_NN_model = adjustable_FDR(input_size = n_features, hidden_sizes = [50, 50, 50], output_size = n_outputs, 
                          target_col = target_col, fix_col = fix_col, projection = False, 
                          Gamma = K, train_adversarially = True, budget_constraint = 'equality')

        optimizer = torch.optim.Adam(fdr_lin_NN_model.parameters(), lr = 1e-3, weight_decay = 1e-5)
        
        # Warm-start using nominal model
        fdr_lin_NN_model.load_state_dict(mlp_model.state_dict(), strict = False)

        # fdr_lin_NN_model.linear_correction_layer.weight.data = mlp_model.model[0].weight.data.clone()
        # fdr_lin_NN_model.linear_correction_layer.bias.data = mlp_model.model[0].bias.data.clone()
        # for j in range(2, 3*2+2, 2):
        #     fdr_lin_NN_model.model[j-1].weight.data = mlp_model.model[j].weight.data.clone()
        #     fdr_lin_NN_model.model[j-1].bias.data = mlp_model.model[j].bias.data.clone()
        
        # if K == 0: 
        #     FDR_Lin_NN_models.append(fdr_lin_NN_model)
        #     continue

        if K > 0:
            fdr_lin_NN_model.adversarial_train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                              patience = patience, freeze_weights = False, attack_type = 'random_sample')
                    
        FDR_Lin_NN_models.append(fdr_lin_NN_model)
        
    if config['save']:
        with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FDR_Lin_NN_models.pickle', 'wb') as handle:
            pickle.dump(FDR_Lin_NN_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:    
    with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FDR_Lin_NN_models.pickle', 'rb') as handle:    
            FDR_Lin_NN_models = pickle.load(handle)
            
#% Retrain without missing features (Tawn, Browell)

# if config['retrain']:
#     retrain_models, retrain_pred, retrain_comb = retrain_model(trainPred, trainY, testPred, 
#                                                            target_col, fix_col, max(K_parameter), base_loss='l2')
#%% Testing: varying the number of missing observations/ persistence imputation
from utility_functions import *


n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
error_metric = 'rmse'
park_ids = list(power_df.columns.values)
K_parameter = np.arange(0, len(target_pred)+1)

percentage = [0, .001, .005, .01, .05, .1]
# percentage = [0, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['Pers', 'LS', 'Lasso', 'Ridge', 'LAD', 'NN', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LAD'] + [f'FinAd-LS-{n_splits}' for n_splits in Max_number_splits] \
+ ['FinAd-NN', 'FDR-NN', 'FDR-Lin-NN']

models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
                    'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
                    'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$',
                    'FDRR-R':'$\mathtt{FA(const, fixed)}$',
                    'LinAdj-FDR':'$\mathtt{FA(linear, fixed)}$',
                    'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 'FinAd-LS-10':'$\mathtt{FA(linear, greedy)}$', 
                    'FinAd-NN':'$\mathtt{FinAd-NN}$'}


mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
n_series = 4
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0

#series_missing = [c + str('_1') for c in park_ids]
#series_missing_col = [pred_col.index(series) for series in series_missing]

# Park IDs for series that could go missing
series_missing = park_ids

imputation = 'persistence'
mean_imput_values = trainPred.mean(0)
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()

config['pattern'] = 'MCAR'

config['save'] = True

for perc in percentage:
    if (config['pattern'] == 'MNAR')and(run_counter>1):
        continue

    for iter_ in range(iterations):
        
        # Dataframe to store predictions
        # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
        temp_Predictions = pd.DataFrame(data = [], columns = models)

        # Initialize dataframe to store results
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]
        
        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(testPred), len(park_ids)))
        if config['pattern'] == 'MNAR':
            P = np.array([[.999, .001], [0.2, 0.8]])
            for j, series in enumerate(series_missing):                
                # Data is MNAR, set values, control the rest within the function 
                miss_ind[:,j] = make_MNAR_chain(P, 0, len(testPred), power_df.copy()[series][split:end].values)

        else:
            P = np.array([[1-perc, perc], [0.2, 0.8]])
            for j in range(len(series_missing)):
                miss_ind[:,j] = make_chain(P, 0, len(testPred))
                #miss_ind[1:,j+1] = miss_ind[:-1,j]
                #miss_ind[1:,j+2] = miss_ind[:-1,j+1]
        
        mask_ind = miss_ind==1
        
        if run_counter%iterations==0: print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
        
        # Predictors w missing values
        miss_X = power_df[split:end].copy()[park_ids]
        miss_X[mask_ind] = np.nan
        
        miss_X = create_feat_matrix(miss_X, config['min_lag'], config['max_lag'])
        
        final_mask_ind = (miss_X.isna().values).astype(int)
        # Predictors w missing values
        miss_X_zero = miss_X.copy()
        miss_X_zero = miss_X_zero.fillna(0)
        
        # Predictors w mean imputation
        if config['impute'] != True:
            imp_X = miss_X_zero.copy()
        else:
            imp_X = miss_X.copy()
            # imputation with persistence or mean            
            if imputation == 'persistence':
                imp_X = miss_X.copy()
                # forward fill == imputation with persistence
                imp_X = imp_X.fillna(method = 'ffill')
                # fill initial missing values with previous data
                for c in imp_X.columns:
                    imp_X[c].loc[imp_X[c].isna()] = trainPred[c].mean()
                
                #for j in series_missing:
                #    imp_X[mask_ind[:,j],j] = imp_X[mask_ind[:,j],j+1]
                    
            elif imputation == 'mean':
                for j in range(imp_X.shape[1]):
                    imp_X[np.where(miss_ind[:,j] == 1), j] = mean_imput_values[j]
        
        
        #### Persistence
        pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
        temp_Predictions['Pers'] = pers_pred.reshape(-1)
        
        # if config['scale']:
        #     pers_pred = target_scaler.inverse_transform(pers_pred)            
        # pers_mae = eval_predictions(pers_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### LS model
        lr_pred = projection(lr.predict(imp_X).reshape(-1,1))
        temp_Predictions['LS'] = lr_pred.reshape(-1)
        
        # if config['scale']:
        #     lr_pred = target_scaler.inverse_transform(lr_pred)            
        # lr_mae = eval_predictions(lr_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### LASSO
        lasso_pred = projection(lasso.predict(imp_X).reshape(-1,1))
        temp_Predictions['Lasso'] = lasso_pred.reshape(-1)

        # if config['scale']:
        #     lasso_pred = target_scaler.inverse_transform(lasso_pred)    
        # lasso_mae = eval_predictions(lasso_pred, Target.values, metric= error_metric)
    
        #### RIDGE
        l2_pred = projection(ridge.predict(imp_X).reshape(-1,1))
        temp_Predictions['Ridge'] = l2_pred.reshape(-1)
        
        # if config['scale']:
        #     l2_pred = target_scaler.inverse_transform(l2_pred)    
        # l2_mae = eval_predictions(l2_pred, Target.values, metric= error_metric)
    
        #### LAD model
        lad_pred = projection(lad.predict(imp_X).reshape(-1,1))
        temp_Predictions['LAD'] = lad_pred.reshape(-1)

        # if config['scale']:
        #     lad_pred = target_scaler.inverse_transform(lad_pred)            
        # lad_mae = eval_predictions(lad_pred.reshape(-1,1), Target.values, metric=error_metric)

        #### LAD-l1 model
        # lad_l1_pred = projection(lad_l1.predict(imp_X).reshape(-1,1))
        # if config['scale']:
        #     lad_l1_pred = target_scaler.inverse_transform(lad_l1_pred)            
        # lad_l1_mae = eval_predictions(lad_l1_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### MLPimp
        mlp_pred = mlp_model.predict(torch.FloatTensor(imp_X.values)).reshape(-1,1)
        temp_Predictions['NN'] = mlp_pred.reshape(-1)

        # if config['scale']:
        #     mlp_pred = target_scaler.inverse_transform(mlp_pred)    
        # mlp_mae = eval_predictions(mlp_pred, Target.values, metric= error_metric)

        #### Adversarial MLP
        # res_mlp_pred = adj_fdr_model.predict(torch.FloatTensor(miss_X_zero.values), torch.FloatTensor(final_mask_ind)).reshape(-1,1)
        # if config['scale']:
        #     res_mlp_pred = target_scaler.inverse_transform(res_mlp_pred)    
        # res_mlp_mae = eval_predictions(res_mlp_pred, Target.values, metric= error_metric)
        
        #### Finite Adaptability - Fixed Partition models: find which model to use
        # col_index = np.zeros(len(Target))
        # if (perc>0)or(config['pattern']=='MNAR'):
        #     for j, ind in enumerate(mask_ind):
        #         n_miss_feat = miss_X.isna().values[j].sum()
        #         col_index[j] = n_miss_feat
                            
        #### FA models with fixed partitions
        #### FDRR-R (select the appropriate model for each case)
        fdr_aar_predictions = []
        for i, k in enumerate(K_parameter):
            
            fdr_pred = gd_FDRR_R_models[i].predict(torch.FloatTensor(miss_X_zero.values)).reshape(-1,1)
            # fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            fdr_pred = projection(fdr_pred)
                
            # Robust
            # if config['scale']: fdr_pred = target_scaler.inverse_transform(fdr_pred)
            fdr_aar_predictions.append(fdr_pred.reshape(-1))
        fdr_aar_predictions = np.array(fdr_aar_predictions).T
        
        # Use only the model with the appropriate K
        final_fdr_aar_pred = fdr_aar_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_aar_pred[j] = fdr_aar_predictions[j, n_miss_feat]
        final_fdr_aar_pred = final_fdr_aar_pred.reshape(-1,1)
                
        temp_Predictions['FDRR-R'] = final_fdr_aar_pred.reshape(-1)

        #### Linearly Adjustable FDR (select the appropriate model for each case)
        ladj_fdr_predictions = []
        for i, k in enumerate(K_parameter):
            
            ladj_fdr_pred = ladj_FDRR_R_models[i].predict(miss_X_zero.values, final_mask_ind).reshape(-1,1)
            # fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            ladj_fdr_pred = projection(ladj_fdr_pred)
                
            ladj_fdr_predictions.append(ladj_fdr_pred.reshape(-1))

        ladj_fdr_predictions = np.array(ladj_fdr_predictions).T
        
        # Use only the model with the appropriate K
        final_ladj_fdr_pred = ladj_fdr_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_ladj_fdr_pred[j] = ladj_fdr_predictions[j, n_miss_feat]
        final_ladj_fdr_pred = final_ladj_fdr_pred.reshape(-1,1)
        
        temp_Predictions['LinAdj-FDR'] = final_ladj_fdr_pred.reshape(-1)

        #### FDR-NN
        fdr_nn_pred_list = []
        for i, k in enumerate(K_parameter):
            
            fdr_nn_pred = FDR_NN_models[i].predict(miss_X_zero.values).reshape(-1,1)
            fdr_nn_pred = projection(fdr_nn_pred)
                
            fdr_nn_pred_list.append(fdr_nn_pred.reshape(-1))

        fdr_nn_pred_list = np.array(fdr_nn_pred_list).T
        
        # Use only the model with the appropriate K
        final_fdr_nn_pred = fdr_nn_pred_list[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_nn_pred[j] = fdr_nn_pred_list[j, n_miss_feat]
        final_fdr_nn_pred = final_fdr_nn_pred.reshape(-1,1)
        temp_Predictions['FDR-NN'] = final_fdr_nn_pred.reshape(-1)

        #### FDR-Lin-NN
        fdr_lin_nn_pred_list = []
        for i, k in enumerate(K_parameter):
            
            fdr_lin_nn_pred = FDR_Lin_NN_models[i].predict(miss_X_zero.values, final_mask_ind).reshape(-1,1)
            fdr_lin_nn_pred = projection(fdr_lin_nn_pred)
                
            fdr_lin_nn_pred_list.append(fdr_lin_nn_pred.reshape(-1))

        fdr_lin_nn_pred_list = np.array(fdr_lin_nn_pred_list).T
        
        # Use only the model with the appropriate K
        final_fdr_lin_nn_pred = fdr_lin_nn_pred_list[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_lin_nn_pred[j] = fdr_lin_nn_pred_list[j, n_miss_feat]
        final_fdr_lin_nn_pred = final_fdr_lin_nn_pred.reshape(-1,1)
        temp_Predictions['FDR-Lin-NN'] = final_fdr_lin_nn_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        fin_LAD_pred = fin_LAD_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        fin_LAD_pred = projection(fin_LAD_pred)
        temp_Predictions['FinAd-LAD'] = fin_LAD_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        fin_NN_pred = fin_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        fin_NN_pred = projection(fin_NN_pred)
        temp_Predictions['FinAd-NN'] = fin_NN_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        for number_splits in Max_number_splits:
            
            fin_LS_pred = fin_LS_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            fin_LS_pred = projection(fin_LS_pred)
            temp_Predictions[f'FinAd-LS-{number_splits}'] = fin_LS_pred.reshape(-1)

        # fin_LS_pred = fin_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # fin_LS_pred = projection(fin_LS_pred)
        # temp_Predictions['FinAd-LS'] = fin_LS_pred.reshape(-1)
        
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values, Target.values)]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values, Target.values)]
        rmse_df = pd.concat([rmse_df, temp_df])
        
        run_counter += 1

pattern = config['pattern']

if config['save']:
    mae_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_MAE_results.csv')
    rmse_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_RMSE_results.csv')
    
#%% Plotting

models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
                    'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
                    'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$','FDRR-R':'$\mathtt{FDR-LS}$',
                    'LinAdj-FDR':'$\mathtt{FDR-Lin-LS}$',
                    'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 'FinAd-LS-10':'$\mathtt{FinAd-LS}$', 
                    'FinAd-LS-25':'$\mathtt{FinAd-LS}$', 'FinAd-LS-5':'$\mathtt{FinAd-LS}$', 
                    'FDR-NN':'FDR-NN'}

 
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

# models_to_plot = ['Pers', 'LS', 'Lasso', 'Ridge', 'LAD', 'NN', 'FDRR-R', 'FinAd-LAD', 'FinAd-LS']
models_to_plot = ['LS', 'LAD','NN', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LAD', 'FinAd-LS-10', 'FinAd-NN', 'FDR-NN', 'FDR-Lin-NN']
marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = mae_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    x_val = temp_df['percentage'].unique().astype(float)
    #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
    plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
             label = m, color = colors[i], marker = marker[i])
    #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
    #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
plt.legend()
plt.ylabel('MAE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
# plt.savefig(f'{cd}//plots//{target_park}_MAE.pdf')
plt.show()
#%%

fig, ax = plt.subplots(constrained_layout = True)
models_to_plot = ['LS','NN', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LS-10', 'FDR-NN', 'FDR-Lin-NN']

temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    x_val = temp_df['percentage'].unique().astype(float)
    #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
    plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
             label = m, color = colors[i], marker = marker[i])
    #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
    #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
# plt.savefig(f'{cd}//plots//{target_park}_RMSE.pdf')
plt.show()

#%% Plot for a single method


 
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['LS', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LS']
models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FDRR-R':'$\mathtt{FinAd(static, fixed)}$',
                    'LinAdj-FDR':'$\mathtt{FinAd(linear, fixed)}$',
                    'FinAd-LS':'$\mathtt{FinAd(linear, greedy)}$'}

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models):    
    if m not in models_to_plot: continue
    else:
        y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
        x_val = temp_df['percentage'].unique().astype(float)
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
                 label = models_to_labels[m], color = colors[i], marker = marker[i])
        #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
        #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
# plt.savefig(f'{cd}//plots//{target_park}_LS_RMSE.pdf')
plt.show()

