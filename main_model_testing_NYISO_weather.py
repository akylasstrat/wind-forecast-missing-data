# -*- coding: utf-8 -*-
"""
Testing NYISO data with weather features

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
    params['start_date'] = '2018-01-01' # start of train set
    params['split_date'] = '2018-06-01' # end of train set/start of test set
    params['end_date'] = '2019-01-01'# end of test set
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 2 # per pair of (n_nodes,percentage)
    # params['pattern'] = 'MCAR'
    params['retrain'] = False
    
    params['min_lag'] = 1
    params['max_lag'] = 2 + params['min_lag']
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)

    params['save'] = False # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv(f'{cd}\\data\\2018_wind_site_5min.csv', index_col = 0, parse_dates=True)
metadata_df = pd.read_csv(f'{cd}\\data\\wind_meta.csv', index_col = 0)

#%%
freq = '15min'
target_park = 'Noble Clinton'
config['min_lag'] = 24
config['save'] = False

id_forecasts_df = pd.read_csv(f'{cd}\\data\\{target_park}_intraday_forecasts_2018.csv', index_col = 0, parse_dates = True)
id_forecasts_df = id_forecasts_df.resample(freq).interpolate()

config['split_date'] = '2018-06-01' # end of train set/start of test set
config['end_date'] = '2019-01-01'

target_zone = metadata_df.loc[target_park].load_zone
power_df = power_df.resample(freq).mean()

scaled_power_df = power_df.copy()

for c in scaled_power_df.columns:
    scaled_power_df[c] = power_df[c].values/metadata_df.loc[c].capacity
    
#%%
# scale between [0,1]/ or divide by total capacity

# Select zone

plant_ids = list(metadata_df[metadata_df['load_zone']==target_zone].index)

print('Number of plants per zone')
print(metadata_df.groupby(['load_zone'])['config'].count())

fig, ax = plt.subplots(constrained_layout = True)
metadata_df.plot(kind='scatter', x = 'longitude', y = 'latitude', ax = ax)
plt.show()

#%%

print(f'target_park:{target_park}')
# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include

config['max_lag'] = 3 + config['min_lag']
# config['pattern'] = 'MNAR'

iterations = 5
percentage = [0, .001, .005, .01, .05, .1]

min_lag = config['min_lag']
max_lag = config['max_lag']

Y, Predictors, pred_col = create_IDsupervised(target_park, scaled_power_df[plant_ids], min_lag, max_lag)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

start = config['start_date']
split = config['split_date']
end = config['end_date']

# trainPred = Predictors[start:split].dropna()
# testPred = Predictors[split:end].dropna()

# Predictors with weather
trainPred = pd.merge(Predictors[start:split], id_forecasts_df[start:split], how='inner', left_index=True, right_index=True).dropna()
testPred = pd.merge(Predictors[split:end], id_forecasts_df[split:end], how='inner', left_index=True, right_index=True).dropna()

trainY = Y[trainPred.index[0]:trainPred.index[-1]].values
testY = Y[testPred.index[0]:testPred.index[-1]].values
Target = Y[testPred.index[0]:testPred.index[-1]]

target_pred = Predictors.columns
fixed_pred = id_forecasts_df.columns
target_col = [np.where(trainPred.columns == c)[0][0] for c in target_pred]
fix_col = [np.where(trainPred.columns == c)[0][0] for c in fixed_pred]

#%% Load trained models

# Load nominal models
with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_LR_weather.pickle', 'rb') as handle:
    lr_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_LAD_weather.pickle', 'rb') as handle:
    lad_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_Ridge_weather.pickle', 'rb') as handle:
    ridge_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_Lasso_weather.pickle', 'rb') as handle:
    lasso_model = pickle.load(handle)

with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_MLP_weather.pickle', 'rb') as handle:
    mlp_model = pickle.load(handle)

# Load adversarial models

with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict_weather.pickle', 'rb') as handle:
    FA_lin_greedy_LS_models_dict = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'rb') as handle:
#     FA_lin_greedy_LS_models_dict = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_greedy_LS_model.pickle', 'rb') as handle:
#     FA_greedy_LS_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_greedy_NN_model.pickle', 'rb') as handle:
#     FA_greedy_NN_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_greedy_NN_model.pickle', 'rb') as handle:    
#     FA_lin_greedy_NN_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_fixed_LS_model.pickle', 'rb') as handle:    
#         FA_fixed_LS_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_fixed_NN_model.pickle', 'rb') as handle:    
#         FA_fixed_NN_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_fixed_LS_model.pickle', 'rb') as handle:    
#         FA_lin_fixed_LS_model = pickle.load(handle)

# with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_fixed_NN_model.pickle', 'rb') as handle:    
#         FA_lin_fixed_NN_model = pickle.load(handle)

#%% Test models
target_col = trainPred.columns
fix_col = []
n_feat = len(target_col)
n_test_obs = len(testY)

error_metric = 'rmse'
# # transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['Pers', 'LS', 'Lasso', 'Ridge', 'LAD', 'NN'] \
    +['FA-greedy-LAD', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS']  + ['FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN','FA-lin-greedy-NN', 'v2FA-lin-fixed-NN']\
    +[f'FA-lin-greedy-LS-{n_splits}' for n_splits in FA_lin_greedy_LS_models_dict.keys()] \

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

#series_missing = [c + str('_1') for c in plant_ids]
#series_missing_col = [pred_col.index(series) for series in series_missing]

# Park IDs for series that could go missing
series_missing = plant_ids

imputation = 'persistence'
mean_imput_values = trainPred.mean(0)
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()


print('Test for MCAR mechanism')
for perc in percentage:
    for iter_ in range(iterations):
        
        torch.manual_seed(run_counter)
        # Dataframe to store predictions
        # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
        temp_Predictions = pd.DataFrame(data = [], columns = models)

        # Initialize dataframe to store results
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]
        
        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(testPred), len(plant_ids)))
        
        # if pattern in ['MNAR', 'MNAR_sq']:
        #     P = np.array([[.999, .001], [0.2, 0.8]])
        #     for j, series in enumerate(series_missing): 
        #         if series == target_park:
        #             # Data is MNAR, set values, control the rest within the function 
        #             miss_ind[:,j] = make_MNAR_chain(P, 0, len(testPred), scaled_power_df.copy()[series][split:end].values, pattern)
        #         else:
        #             P = np.array([[1-0.05, 0.05], [0.2, 0.8]])
        #             miss_ind[:,j] = make_chain(P, 0, len(testPred))
                    
        # elif pattern == 'MCAR':
        if freq in ['15min', '5min']:
            P = np.array([[1-perc, perc], [0.1, 0.9]])
        elif freq in ['30min']:
            P = np.array([[1-perc, perc], [0.2, 0.8]])

        for j in range(len(series_missing)):
            miss_ind[:,j] = make_chain(P, 0, len(testPred))
            #miss_ind[1:,j+1] = miss_ind[:-1,j]
            #miss_ind[1:,j+2] = miss_ind[:-1,j+1]
        
        mask_ind = miss_ind==1
        
        if run_counter%iterations==0: print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
        
        # Predictors w missing values
        miss_X = scaled_power_df[split:end].copy()[plant_ids]
        miss_X = miss_X[testPred.index[0]:testPred.index[-1]]
        miss_X[mask_ind] = np.nan
        
        miss_X = create_feat_matrix(miss_X, config['min_lag'], config['max_lag'])
        
        # Add weather features 
        miss_X = pd.merge(miss_X, id_forecasts_df[split:end], how='inner', left_index=True, right_index=True)

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
                # imp_X = imp_X.fillna(method = 'ffill')
                imp_X = imp_X.ffill()
                
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
        # FA_fixed_LS_pred = FA_fixed_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_fixed_LS_pred = projection(FA_fixed_LS_pred)
        # temp_Predictions['FA-fixed-LS'] = FA_fixed_LS_pred.reshape(-1)

        ## NN model
        # FA_fixed_NN_pred = FA_fixed_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_fixed_NN_pred = projection(FA_fixed_NN_pred)
        # temp_Predictions['FA-fixed-NN'] = FA_fixed_NN_pred.reshape(-1)

        #### Finite Adaptability - Linear - Fixed Partitions
        ## LS model
        # FA_lin_fixed_LS_pred = FA_lin_fixed_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_lin_fixed_LS_pred = projection(FA_lin_fixed_LS_pred)
        # temp_Predictions['FA-lin-fixed-LS'] = FA_lin_fixed_LS_pred.reshape(-1)

        ## NN model
        # FA_lin_fixed_NN_pred = FA_lin_fixed_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_lin_fixed_NN_pred = projection(FA_lin_fixed_NN_pred)
        # temp_Predictions['FA-lin-fixed-NN'] = FA_lin_fixed_NN_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS        
        # FA_lin_greedy_NN_pred = FA_lin_greedy_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_lin_greedy_NN_pred = projection(FA_lin_greedy_NN_pred)
        # temp_Predictions['FA-lin-greedy-NN'] = FA_lin_greedy_NN_pred.reshape(-1)

        #### FA-Fixed-LS and NN
        # FA_greedy_LS_pred = FA_greedy_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_greedy_LS_pred = projection(FA_greedy_LS_pred)
        # temp_Predictions['FA-greedy-LS'] = FA_greedy_LS_pred.reshape(-1)

        # FA_greedy_NN_pred = FA_greedy_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # FA_greedy_NN_pred = projection(FA_greedy_NN_pred)
        # temp_Predictions['FA-greedy-NN'] = FA_greedy_NN_pred.reshape(-1)

        #### FINITE-RETRAIN-LAD and LS
        for number_splits in FA_lin_greedy_LS_models_dict.keys():
            
            FA_lin_greedy_LS_pred = FA_lin_greedy_LS_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            FA_lin_greedy_LS_pred = projection(FA_lin_greedy_LS_pred)
            temp_Predictions[f'FA-lin-greedy-LS-{number_splits}'] = FA_lin_greedy_LS_pred.reshape(-1)
        
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values[max_lag-1:], Target.values[max_lag-1:])]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values[max_lag-1:], Target.values[max_lag-1:])]
        rmse_df = pd.concat([rmse_df, temp_df])
        
        run_counter += 1

    if config['save']:
        mae_df.to_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results.csv')
        rmse_df.to_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results.csv')
        #%%
    ls_models = ['LS', 'FA-greedy-LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-lin-greedy-LS-10', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5']
    rmse_df.groupby(['percentage']).mean()[ls_models].plot()
    #%%
    nn_models = ['NN', 'FA-greedy-NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-lin-greedy-NN']
    rmse_df.groupby(['percentage']).mean()[nn_models].plot()


stop_here
#%% Test for MNAR missing data

mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0

# Park IDs for series that could go missing
series_missing = plant_ids

imputation = 'persistence'
mean_imput_values = trainPred.mean(0)
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()


print('Test for MNAR mechanism')
for iter_ in range(iterations):
    print(f'Iteration:{iter_}')
    torch.manual_seed(run_counter)
    # Dataframe to store predictions
    # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
    temp_Predictions = pd.DataFrame(data = [], columns = models)

    # Initialize dataframe to store results
    temp_df = pd.DataFrame()
    temp_df['percentage'] = [perc]
    temp_df['iteration'] = [iter_]
    
    # generate missing data
    miss_ind = np.zeros((len(testPred), len(plant_ids)))
    
    if freq in ['15min', '5min']:
        P_init = np.array([[.999, .001], [0.2, 0.8]])
        P_norm = np.array([[1-0.01, 0.01], [0.1, 0.9]])
    elif freq in ['30min']:
        P = np.array([[.999, .001], [0.2, 0.8]])
        P_norm = np.array([[1-0.05, 0.05], [0.2, 0.8]])
            
    for j, series in enumerate(series_missing): 
        if series == target_park:
            # Data is MNAR, set values, control the rest within the function 
            miss_ind[:,j] = make_MNAR_chain(P, 0, len(testPred), scaled_power_df.copy()[series][split:end].values, 'MNAR')
        else:
            
            miss_ind[:,j] = make_chain(P_norm, 0, len(testPred))
    
    mask_ind = miss_ind==1
    
    if run_counter%iterations==0: print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
    
    # Predictors w missing values
    miss_X = scaled_power_df[split:end].copy()[plant_ids]
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
            # imp_X = imp_X.fillna(method = 'ffill')
            imp_X = imp_X.ffill()
            
            # fill initial missing values with previous data
            for c in imp_X.columns:
                imp_X[c].loc[imp_X[c].isna()] = trainPred[c].mean()
                            
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

if config['save']:
    mae_df.to_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_MAE_results.csv')
    rmse_df.to_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_RMSE_results.csv')
    
ls_models = ['LS', 'FA-greedy-LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-lin-greedy-LS-10']
rmse_df.groupby(['percentage']).mean()[ls_models].plot()
#%%
nn_models = ['NN', 'FA-greedy-NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-lin-greedy-NN']
rmse_df.groupby(['percentage']).mean()[nn_models].plot()

