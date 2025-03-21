# -*- coding: utf-8 -*-
"""
Sensitivity analysis

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
config['min_lag'] = 2
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

#%% Finite adaptability with MLPs

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

config['train'] = True
config['save'] = False

Total_number_splits = [1, 2, 5, 10, 25]

fin_LS_model_dict = {}

if config['train']:
    
    for number_splits in Total_number_splits:
        
        fin_LS_model = FiniteLinear_MLP(target_col = target_col, fix_col = fix_col, Max_models = number_splits, D = 1_000, red_threshold = 1e-5, 
                                                    input_size = n_features, hidden_sizes = [], output_size = n_outputs, projection = True, 
                                                    train_adversarially = True, budget_constraint = 'inequality', attack_type = 'greedy')
        
        fin_LS_model.fit(trainPred.values, trainY, val_split = 0.0, tree_grow_algo = 'leaf-wise', max_gap = 0.05, 
                              epochs = num_epochs, patience = patience, verbose = 0, optimizer = 'Adam', 
                             lr = 1e-3, batch_size = batch_size)
        
        fin_LS_model_dict[number_splits] = fin_LS_model
#     if config['save']:
#         with open(f'{cd}\\trained-models\\{target_park}_fin_LS_model_{number_splits}.pickle', 'wb') as handle:
#             pickle.dump(fin_LS_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
# else:
#     with open(f'{cd}\\trained-models\\{target_park}_fin_LS_model.pickle', 'rb') as handle:    
#             fin_LS_model = pickle.load(handle)
#

#%% Testing: varying the number of missing observations/ persistence imputation
from utility_functions import *

n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
error_metric = 'rmse'
park_ids = list(power_df.columns.values)
K_parameter = np.arange(0, len(target_pred)+1)

percentage = [0, .001, .005, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['Pers', 'LS'] + [f'FinAd-LS-{n_splits}' for n_splits in Total_number_splits]
# models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
#                     'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
#                     'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$','FDRR-R':'$\mathtt{FDRR-R}$',
#                     'LinAdj-FDR':'$\mathtt{LinAdj-FDR}$',
#                     'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 'FinAd-LS':'$\mathtt{FinAd-LS}$'}


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

        #### FINITE-RETRAIN-LAD and LS
        for number_splits in Total_number_splits:
            
            fin_LS_pred = fin_LS_model_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            fin_LS_pred = projection(fin_LS_pred)
            temp_Predictions[f'FinAd-LS-{number_splits}'] = fin_LS_pred.reshape(-1)
            
        
            
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values, Target.values)]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values, Target.values)]
        rmse_df = pd.concat([rmse_df, temp_df])
        
        run_counter += 1

pattern = config['pattern']

# if config['save']:
#     mae_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_MAE_results.csv')
#     rmse_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_RMSE_results.csv')
    
#%% Plotting

models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
                    'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
                    'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$','FDRR-R':'$\mathtt{FDR-LS}$',
                    'LinAdj-FDR':'$\mathtt{FDR-Lin-LS}$',
                    'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 
                    'FinAd-LS-1':'1', 'FinAd-LS-2':'2', 'FinAd-LS-5':'5', 
                    'FinAd-LS-10':'10', 'FinAd-LS-25':'25'}

 

models_to_plot = models[1:]

color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

plt.plot(temp_df['percentage'].unique().astype(float), 100*temp_df.groupby(['percentage'])['LS'].mean().values, 
         linestyle = '--', color = 'black', label = models_to_labels['LS'])

for i, m in enumerate([f'FinAd-LS-{n_splits}' for n_splits in Total_number_splits]):    
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
plt.savefig(f'{cd}//plots//{target_park}_sensitivity_RMSE.pdf')
plt.show()