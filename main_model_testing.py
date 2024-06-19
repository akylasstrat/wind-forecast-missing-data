# -*- coding: utf-8 -*-
"""
Model testing on missing data

@author: a.stratigakos
"""

import pickle
import os, sys
# import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import time
import itertools
# import random

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV


from utility_functions import * 
from FDR_regressor import *
from QR_regressor import *
from torch_custom_layers import *
from finite_adaptability_model_functions import *

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
    params['scale'] = False
    params['impute'] = True # If True, apply mean imputation for missing features
 
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
    
    params['min_lag'] = 1
    params['max_lag'] = 2 + params['min_lag']
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)

    params['save'] = True # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0)
metadata_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_metadata.csv', index_col=0)

# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)

#%%
target_park = 'p_1088'

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
config['min_lag'] = 1
config['max_lag'] = 2 + config['min_lag']
config['pattern'] = 'MCAR'
config['save'] = True
iterations = 5
percentage = [0, .001, .005, .01, .05, .1]

min_lag = config['min_lag']
max_lag = config['max_lag']

power_df = power_df
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)

start = config['start_date']
split = config['split_date']
end = config['end_date']

trainY = Y[start:split].values
testY = Y[split:end].values
Target = Y[split:end]

trainPred = Predictors[start:split]
testPred = Predictors[split:end]

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

#%% Load trained models

# Load nominal models
with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_LR.pickle', 'rb') as handle:
    lr_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_LAD.pickle', 'rb') as handle:
    lad_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_Ridge.pickle', 'rb') as handle:
    ridge_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_Lasso.pickle', 'rb') as handle:
    lasso_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_MLP.pickle', 'rb') as handle:
    mlp_model = pickle.load(handle)

# Load adversarial models
with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'rb') as handle:
    FA_lin_greedy_LS_models_dict = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LAD_model.pickle', 'rb') as handle:    
    FA_greedy_LAD_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'rb') as handle:
    FA_lin_greedy_LS_models_dict = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_LS_model.pickle', 'rb') as handle:
    FA_greedy_LS_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_greedy_NN_model.pickle', 'rb') as handle:
    FA_greedy_NN_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_greedy_NN_model.pickle', 'rb') as handle:    
    FA_lin_greedy_NN_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_LS_model.pickle', 'rb') as handle:    
        FA_fixed_LS_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_fixed_NN_model.pickle', 'rb') as handle:    
        FA_fixed_NN_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_LS_model.pickle', 'rb') as handle:    
        FA_lin_fixed_LS_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\{min_lag}_steps\\{target_park}_FA_lin_fixed_NN_model.pickle', 'rb') as handle:    
        FA_lin_fixed_NN_model = pickle.load(handle)


#%% Test models

n_feat = len(target_col)
n_test_obs = len(testY)

error_metric = 'rmse'
park_ids = list(power_df.columns.values)
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['Pers', 'LS', 'Lasso', 'Ridge', 'LAD', 'NN'] \
    +['FA-greedy-LAD', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS'] + [f'FA-lin-greedy-LS-{n_splits}' for n_splits in FA_lin_greedy_LS_models_dict.keys()] \
    + ['FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN','FA-lin-greedy-NN']

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

for perc in percentage:
    if (config['pattern'] == 'MNAR')and(run_counter>1):
        continue
    for iter_ in range(iterations):
        
        torch.manual_seed(run_counter)
        run_counter += 1
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
        
        
        ############ Impute-then-Regress
        
        #### Persistence
        pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
        temp_Predictions['Pers'] = pers_pred.reshape(-1)
                
        #### LS model
        lr_pred = projection(lr_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['LS'] = lr_pred.reshape(-1)
                
        #### LASSO
        lasso_pred = projection(lasso_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['Lasso'] = lasso_pred.reshape(-1)
    
        #### RIDGE
        l2_pred = projection(ridge_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['Ridge'] = l2_pred.reshape(-1)
            
        #### LAD model
        lad_pred = projection(lad_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['LAD'] = lad_pred.reshape(-1)

        #### MLPimp
        mlp_pred = mlp_model.predict(torch.FloatTensor(imp_X.values)).reshape(-1,1)
        temp_Predictions['NN'] = mlp_pred.reshape(-1)
                
        ######### Adversarial Models
        
        #### Finite Adaptability - Fixed Partitions
        ## LS model
        FA_fixed_LS_pred = FA_fixed_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_fixed_LS_pred = projection(FA_fixed_LS_pred)
        temp_Predictions['FA-fixed-LS'] = FA_fixed_LS_pred.reshape(-1)

        ## NN model
        FA_fixed_NN_pred = FA_fixed_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_fixed_NN_pred = projection(FA_fixed_NN_pred)
        temp_Predictions['FA-fixed-NN'] = FA_fixed_NN_pred.reshape(-1)

        #### Finite Adaptability - Linear - Fixed Partitions
        ## LS model
        FA_lin_fixed_LS_pred = FA_lin_fixed_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_lin_fixed_LS_pred = projection(FA_lin_fixed_LS_pred)
        temp_Predictions['FA-lin-fixed-LS'] = FA_lin_fixed_LS_pred.reshape(-1)

        ## NN model
        FA_lin_fixed_NN_pred = FA_lin_fixed_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_lin_fixed_NN_pred = projection(FA_lin_fixed_NN_pred)
        temp_Predictions['FA-lin-fixed-NN'] = FA_lin_fixed_NN_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        FA_greedy_LAD_pred = FA_greedy_LAD_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_greedy_LAD_pred = projection(FA_greedy_LAD_pred)
        temp_Predictions['FA-greedy-LAD'] = FA_greedy_LAD_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        
        FA_lin_greedy_NN_pred = FA_lin_greedy_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_lin_greedy_NN_pred = projection(FA_lin_greedy_NN_pred)
        temp_Predictions['FA-lin-greedy-NN'] = FA_lin_greedy_NN_pred.reshape(-1)

        #### FA-Fixed-LS and NN
        FA_greedy_LS_pred = FA_greedy_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_greedy_LS_pred = projection(FA_greedy_LS_pred)
        temp_Predictions['FA-greedy-LS'] = FA_greedy_LS_pred.reshape(-1)

        FA_greedy_NN_pred = FA_greedy_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        FA_greedy_NN_pred = projection(FA_greedy_NN_pred)
        temp_Predictions['FA-greedy-NN'] = FA_greedy_NN_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        for number_splits in FA_lin_greedy_LS_models_dict.keys():
            
            FA_lin_greedy_LS_pred = FA_lin_greedy_LS_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            FA_lin_greedy_LS_pred = projection(FA_lin_greedy_LS_pred)
            temp_Predictions[f'FA-lin-greedy-LS-{number_splits}'] = FA_lin_greedy_LS_pred.reshape(-1)
        
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