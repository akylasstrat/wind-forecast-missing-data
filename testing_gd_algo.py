# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:27:37 2024

@author: astratig
"""

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
    ''' Retrain model withtout missing features
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
        for lag in range(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    return Predictors

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['scale'] = False
    params['train'] = False # If True, then train models, else tries to load previous runs
    params['save'] = False # If True, then saves models and results
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
target_park = 'p_1257'

# number of lags back to consider
min_lag = config['min_lag']
config['max_lag'] = 2
max_lag = config['max_lag']

power_df = power_df
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = '2019-01-01'
split = '2019-03-01'
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


#trainPred = np.column_stack((trainPred, np.ones(len(trainPred))))
#testPred = np.column_stack((testPred, np.ones(len(testPred))))

#%%%% Linear models: linear regression, ridge, lasso 

# Hyperparameter tuning with by cross-validation
param_grid = {"alpha": [10**pow for pow in range(-5,2)]}

ridge = GridSearchCV(Ridge(fit_intercept = True, max_iter = 5000), param_grid)
ridge.fit(trainPred, trainY)

lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 5000), param_grid)
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
persistence_pred = np.insert(persistence_pred, 0, trainY[-1]).reshape(-1,1)
persistence_mae = eval_point_pred(persistence_pred, Target.values, digits=4)[1]

print('Climatology: ', eval_point_pred(trainY.mean(), Target.values, digits=4))
print('Persistence: ', eval_point_pred(persistence_pred, Target.values, digits=4))
print('LR: ', eval_point_pred(lr_pred, Target.values, digits=4))
print('Lasso: ', eval_point_pred(lasso_pred, Target.values, digits=4))
print('Ridge: ', eval_point_pred(ridge_pred, Target.values, digits=4))
print('LAD: ', eval_point_pred(lad_pred, Target.values, digits=4))
print('LAD-L1: ', eval_point_pred(lad_l1_pred, Target.values, digits=4))


#%%

lad = QR_regressor(fit_intercept = True)
lad.fit(trainPred, trainY)

objval = []

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

#%%%% FDDR-AAR: train one model per value of \Gamma

case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_fdrr-aar_{max_lag}.pickle'

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

K_parameter = np.arange(0, len(target_pred)+1)

FDRR_AAR_models = []
config['train'] = True
if config['train']:
    for K in K_parameter:
        print('Gamma: ', K)
        
        fdr = FDR_regressor(K = K)
        fdr.fit(trainPred.values, trainY, target_col, fix_col, verbose=-1, solution = 'reformulation')  
        
        fdr_pred = fdr.predict(testPred).reshape(-1,1)
    
        print('FDR: ', eval_point_pred(fdr_pred, Target.values, digits=2))
        FDRR_AAR_models.append(fdr)
    
else:
    with open(output_file_name, 'rb') as handle:    
            FDRR_AAR_models = pickle.load(handle)
#%%%% Gradient-based FDRR

from torch_custom_layers import * 

batch_size = 1000
num_epochs = 1000
learning_rate = 1e-3
patience = 25
    
# Standard MLPs (separate) forecasting wind production and dispatch decisions
tensor_trainY = torch.FloatTensor(trainY)
tensor_validY = torch.FloatTensor(trainY)
tensor_testY = torch.FloatTensor(testY)

tensor_trainPred = torch.FloatTensor(trainPred.values)
tensor_validPred = torch.FloatTensor(trainPred.values)
tensor_testPred = torch.FloatTensor(testPred.values)

#### MLP model to predict wind from features
train_base_data_loader = create_data_loader([tensor_trainPred, tensor_trainY], batch_size = batch_size, shuffle = False)
valid_base_data_loader = create_data_loader([tensor_validPred, tensor_validY], batch_size = batch_size, shuffle = False)

n_features = tensor_trainPred.shape[1]
n_outputs = tensor_trainY.shape[1]

gd_FDR_models = []

# nominal model (no missing data) to warm-start all future models
nominal_model = gd_FDRR(input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                          target_col = target_col, fix_col = fix_col, projection = False, Gamma = 0, train_adversarially = False)

optimizer = torch.optim.Adam(nominal_model.parameters(), lr = 1e-2)
nominal_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                      patience = patience, verbose = 0)
#%%
batch_size = 500
from torch_custom_layers import *

for K in [10]:
    
    gd_fdr_model = gd_FDRR(input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                              target_col = target_col, fix_col = fix_col, projection = False, 
                              Gamma = K, train_adversarially = True)
    
    optimizer = torch.optim.Adam(gd_fdr_model.parameters(), lr = 1e-2)

    # initialize weights with nominal model
    gd_fdr_model.load_state_dict(nominal_model.state_dict(), strict=False)
        
    gd_fdr_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                          patience = patience, verbose = 0, warm_start = False)

    gd_fdr_pred = gd_fdr_model.predict(tensor_testPred, project = True)
    
    gd_FDR_models.append(gd_fdr_model)
    
    print('GD FDRR: ', eval_point_pred(gd_fdr_pred, Target.values, digits=4))
    print('FDRR: ', eval_point_pred(projection(FDRR_AAR_models[K].predict(testPred).reshape(-1,1)), Target.values, digits=4))
    
    plt.plot(gd_fdr_model.predict(tensor_testPred*(1-best_alpha[0:1]))[:1000], label='GD')
    plt.plot(projection(FDRR_AAR_models[K].predict(testPred*(1-best_alpha.detach().numpy()[0:1]))[:1000]), label='MinMax')
    plt.legend()
    plt.show()

    for name, param in gd_fdr_model.named_parameters():
        if param.requires_grad:
            plt.plot(param.data.detach().numpy().T)
    plt.plot(FDRR_AAR_models[K].coef_)
    plt.show()

#%%

# Version two
batch_size = 50

X = tensor_trainPred
y = tensor_trainY

for K in [10]:
    # find a good vertex for adversarial examples
    y_hat = nominal_model.forward(X)        
    current_loss = nominal_model.estimate_loss(y_hat, y).mean()
    # initialize wc loss and adversarial example
    wc_loss = current_loss.data
    best_alpha =  torch.zeros_like(X)
    current_target_col = torch.tensor(target_col)
        
    # Iterate over gamma, greedily add one feature per iteration
    for g in range(K):
        
        # store losses for all features
        local_loss = []
        # placeholders for splitting a column
        best_col = None
        apply_split = False  
        # !!!!! Important to use clone/copy here, else we update both
        alpha_init = torch.clone(best_alpha)   
        
        # Nominal loss for this iteration using previous alpha values
        y_hat = nominal_model.forward(X*(1-alpha_init))
        temp_nominal_loss = nominal_model.estimate_loss(y_hat, y).mean().data
                
        # loop over target columns (for current node), find worst-case loss:
        for col in current_target_col:
            # create adversarial example
            alpha_temp = torch.clone(alpha_init)
            
            # set feature to missing
            alpha_temp[:,col] = 1
                
            # predict using adversarial example
            y_hat = nominal_model.forward(X*(1-alpha_temp))
            temp_loss = nominal_model.estimate_loss(y_hat, y).mean().data
            local_loss.append(temp_loss.data)

        best_col_ind = np.argsort(local_loss)[-1]                
        wc_loss = np.max(local_loss)
        best_col = current_target_col[best_col_ind]

        if wc_loss > temp_nominal_loss:
            # update
            best_alpha[:,best_col] = 1
            apply_split = True
            
        #update list of eligible columns
        if apply_split:
            current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
        else:
            break
        
    train_base_data_loader = create_data_loader([tensor_trainPred*(1-best_alpha), tensor_trainY], batch_size = batch_size, shuffle = True)
    valid_base_data_loader = create_data_loader([tensor_validPred*(1-best_alpha), tensor_validY], batch_size = batch_size, shuffle = True)
                
    gd_fdr_model = gd_FDRR(input_size = n_features, hidden_sizes = [], output_size = n_outputs, 
                              target_col = target_col, fix_col = fix_col, projection = False, Gamma = K, train_adversarially=False)
    
    optimizer = torch.optim.Adam(gd_fdr_model.parameters(), lr = 1e-3)

    # initialize weights with nominal model
    # gd_fdr_model.load_state_dict(nominal_model.state_dict(), strict=False)

    gd_fdr_model.train_model(train_base_data_loader, valid_base_data_loader, optimizer, epochs = num_epochs, 
                          patience = patience, verbose = 0, warm_start = False)

    gd_fdr_pred = gd_fdr_model.predict(tensor_testPred, project = True)
    
    gd_FDR_models.append(gd_fdr_model)
    
    print('GD FDRR: ', eval_point_pred(gd_fdr_pred, Target.values, digits=4))
    print('FDRR: ', eval_point_pred(projection(FDRR_AAR_models[K].predict(testPred).reshape(-1,1)), Target.values, digits=4))
    
    plt.plot(gd_fdr_model.predict(tensor_testPred*(1-best_alpha[0:1]))[:1000], label='GD')
    plt.plot(projection(FDRR_AAR_models[K].predict(testPred*(1-best_alpha.detach().numpy()[0:1]))[:1000]), label='MinMax')
    plt.legend()
    plt.show()

    for name, param in gd_fdr_model.named_parameters():
        if param.requires_grad:
            plt.plot(param.data.detach().numpy().T)
    plt.plot(FDRR_AAR_models[K].coef_)
    plt.show()


            
#%%%%%%%%% Testing: varying the number of missing observations/ persistence imputation
from utility_functions import *

n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 2
error_metric = 'mae'
park_ids = list(power_df.columns.values)

#percentage = [0, .001, .005, .01, .05, .1]
percentage = [0, .001, .005, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['PERS', 'FDRR-R', 'FDRR-AAR', 'GD-FDRR-AAR', 'RETRAIN', 'FIN-RETRAIN', 'MLP', 
          'resMLP']
labels = ['$\mathtt{PERS}$', '$\mathtt{LS}$', '$\mathtt{LS}_{\ell_2}$', '$\mathtt{LS}_{\ell_1}$',
           '$\mathtt{LAD}$', '$\mathtt{LAD}_{\ell_1}$','$\mathtt{FDRR-R}$', '$\mathtt{FDRR-AAR}$', '$\mathtt{GD-FDRR}$', 
           '$\mathtt{MLP}$', '$\mathtt{resMLP}$']


mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
n_series = 4
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0


#series_missing = [c + str('_1') for c in park_ids]
series_missing = park_ids
#series_missing_col = [pred_col.index(series) for series in series_missing]

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


for perc in percentage:
    if (config['pattern'] == 'MNAR')and(run_counter>1):
        continue

    for iter_ in range(iterations):
        
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
                imp_X = imp_X.fillna(method = 'ffill').fillna(method = 'bfill')
                #for j in series_missing:
                #    imp_X[mask_ind[:,j],j] = imp_X[mask_ind[:,j],j+1]
                    
            elif imputation == 'mean':
                for j in range(imp_X.shape[1]):
                    imp_X[np.where(miss_ind[:,j] == 1), j] = mean_imput_values[j]
        
        # initialize empty dataframe
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]
        
        #### Persistence
        pers_pred = imp_X[f'{target_park}_1'].values.reshape(-1,1)

        if config['scale']:
            pers_pred = target_scaler.inverse_transform(pers_pred)            
        pers_mae = eval_predictions(pers_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### FDRR-AAR (select the appropriate model for each case)
        fdr_aar_predictions = []
        for i, k in enumerate(K_parameter):
            fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            fdr_pred = projection(fdr_pred)
                
            # Robust
            if config['scale']: fdr_pred = target_scaler.inverse_transform(fdr_pred)
            fdr_aar_predictions.append(fdr_pred.reshape(-1))
        fdr_aar_predictions = np.array(fdr_aar_predictions).T

        # Use only the model with the appropriate K
        final_fdr_aar_pred = fdr_aar_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_aar_pred[j] = fdr_aar_predictions[j, n_miss_feat]
        final_fdr_aar_pred = final_fdr_aar_pred.reshape(-1,1)
        
        fdr_aar_mae = eval_predictions(final_fdr_aar_pred, Target.values, metric= error_metric)


        #### FDRR-AAR (select the appropriate model for each case)
        gd_fdr_aar_predictions = []
        for i, k in enumerate(K_parameter):
            
            fdr_pred = gd_FDR_models[i].predict(torch.FloatTensor(miss_X_zero.values)).reshape(-1,1)
                
            # Robust
            if config['scale']: fdr_pred = target_scaler.inverse_transform(fdr_pred)
            gd_fdr_aar_predictions.append(fdr_pred.reshape(-1))
        gd_fdr_aar_predictions = np.array(gd_fdr_aar_predictions).T

        # Use only the model with the appropriate K
        final_gd_fdr_aar_pred = gd_fdr_aar_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_gd_fdr_aar_pred[j] = gd_fdr_aar_predictions[j, n_miss_feat]
        final_gd_fdr_aar_pred = final_gd_fdr_aar_pred.reshape(-1,1)
        
        gd_fdr_aar_mae = eval_predictions(final_gd_fdr_aar_pred, Target.values, metric= error_metric)
        
        temp_df['PERS'] = [pers_mae]                        
        temp_df['FDRR-AAR'] = fdr_aar_mae
        temp_df['GD-FDRR-AAR'] = gd_fdr_aar_mae

        #temp_df['FDRR-CL'] = fdr_cl_mae
        
        mae_df = pd.concat([mae_df, temp_df])
        run_counter += 1
    
if config['save']:
    mae_df.to_csv(f'{cd}\\{case_folder}\\results\\{target_park}_ID_results.csv')
    
# Plotting 
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['PERS', 'FDRR-AAR', 'GD-FDRR-AAR']
marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
labels = ['$\mathtt{PERS}$', '$\mathtt{LS}$', '$\mathtt{LS}_{\ell_2}$', '$\mathtt{LS}_{\ell_1}$',
           '$\mathtt{LAD}$', '$\mathtt{LAD}_{\ell_1}$','$\mathtt{FDRR}$', '$\mathtt{FDRR-AAR}$', '$\mathtt{GD-FDRR}$', '$\mathtt{FIN-RETRAIN}$', 
           '$\mathtt{MLP}$', '$\mathtt{resMLP}$']


ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = mae_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models):    
    if m not in models_to_plot: continue
    else:
        y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
        x_val = temp_df['percentage'].unique().astype(float)
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
                 label = labels[i], color = colors[i], marker = marker[i])
        #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
        #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    

plt.legend()
plt.ylabel('MAE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2)
#ax.set_xscale('log')
plt.show()

