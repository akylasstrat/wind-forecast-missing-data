# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:37:29 2025

@author: a.stratigakos
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from sklearn.preprocessing import MinMaxScaler

from utility_functions import * 
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
    params['scale'] = False
    params['impute'] = True # If True, apply mean imputation for missing features\
 
    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2018-01-01' # start of train set
    params['split_date'] = '2018-06-01' # end of train set/start of test set
    params['end_date'] = '2019-01-01'# end of test set
    
    params['min_lag'] = 1
    params['max_lag'] = 2 + params['min_lag']
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)

    params['save'] = False # If True, then saves models and results
    
    return params

from scipy import stats

def diebold_mariano_test(e1, e2, h=1, power=2):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters:
    - e1, e2: Forecast errors (numpy arrays) from two models
    - h: Forecast horizon (default = 1)
    - power: 1 for MAE, 2 for MSE/RMSE

    Returns:
    - DM statistic
    - p-value (two-tailed)
    """
    # Ensure same length
    e1, e2 = np.array(e1), np.array(e2)
    assert e1.shape == e2.shape, "Forecast error arrays must be the same shape"

    # Calculate loss differentials
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    elif power == 2:
        d = e1**2 - e2**2
    else:
        raise ValueError("Power must be 1 (MAE) or 2 (MSE)")

    d_mean = np.mean(d)
    n = len(d)

    # Newey-West variance estimator with lag = h-1
    def newey_west_variance(d, h):
        n = len(d)
        gamma_0 = np.var(d, ddof=1)
        gamma = [np.cov(d[:-lag], d[lag:])[0,1] for lag in range(1, h)]
        return gamma_0 + 2 * sum((1 - lag/h) * g for lag, g in enumerate(gamma, start=1))

    var_d = newey_west_variance(d, h)
    dm_stat = d_mean / np.sqrt(var_d / n)

    # Two-sided p-value
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))

    return dm_stat, p_value

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv(f'{cd}\\data\\2018_wind_site_5min.csv', index_col = 0, parse_dates=True)
metadata_df = pd.read_csv(f'{cd}\\data\\wind_meta.csv', index_col = 0)

freq = '15min'
target_park = 'Noble Clinton'
horizon = 1

test_MCAR = True # Performs the experiments presented in the paper
config['save'] = False
# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
min_lag = horizon
max_lag = min_lag + 3
trained_models_path = f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps'


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

print(f'target_park:{target_park}')

# iterations = 50
# percentage = [0, .01, .05, .1]

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

with open(f'{trained_models_path}\\{target_park}_LR_weather.pickle', 'rb') as handle:
    lr_model = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_LAD_weather.pickle', 'rb') as handle:
    lad_model = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_Ridge_weather.pickle', 'rb') as handle:
    ridge_model = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_Lasso_weather.pickle', 'rb') as handle:
    lasso_model = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_MLP_weather.pickle', 'rb') as handle:
    mlp_model = pickle.load(handle)

### Load adversarial models


### FA - LEARN - LDR - LR/NN
with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)
    
with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)

### FA - FIXED - LDR - LR/NN
with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LDR_LR_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LDR_LR_model = pickle.load(handle)
    
with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LDR_NN_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LDR_NN_model = pickle.load(handle)

### FA - LEARN - LR/NN
with open(f'{trained_models_path}\\{target_park}_FA_LEARN_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_NN_models_dict = pickle.load(handle)
    
with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)

### FA - FIXED - LR/NN
with open(f'{trained_models_path}\\{target_park}_FA_FIXED_LR_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LR_model = pickle.load(handle)
    
with open(f'{trained_models_path}\\{target_park}_FA_FIXED_NN_model_weather.pickle', 'rb') as handle:
    FA_FIXED_NN_model = pickle.load(handle)


#%% Test models
target_col = trainPred.columns
fix_col = []
n_feat = len(target_col)
n_test_obs = len(testY)

error_metric = 'rmse'
# # transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['LR', 'NN'] \
        + ['FA-LEARN-LDR-LR-10', 'FA-LEARN-LDR-NN-10', 'FA-LEARN-LR-10', 'FA-LEARN-NN-10']
            
# Parameters
iterations = range(2)
Probability_0_1 = [.05, .1, .2]
Probability_1_0 = [1, .2]
# Probability_1_0 = [.1]
num_series = [1]

full_experiment_list = list(itertools.product(iterations, Probability_0_1, Probability_1_0, num_series))

#%%
# Check if there are saved results
mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'P_0_1', 'P_1_0', 'num_series'])
rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'P_0_1', 'P_1_0', 'num_series'])
run_counter = 0

# supress warning
pd.options.mode.chained_assignment = None

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
#%%
rmse_per_missing_df = pd.DataFrame()
number_splits = 10

if test_MCAR:
    print('Test for MCAR mechanism')
    
    for iter_, prob_0_1, prob_1_0, n_miss_series in full_experiment_list:
        
        print(run_counter)
        torch.manual_seed(run_counter)
        
        if (prob_0_1 == 0) and (iter_ >0):      
            temp_mae_df = mae_df.query('P_0_1==0.0 and iteration == 0').iloc[0:1]
            temp_mae_df['iteration'] = iter_                
            mae_df = pd.concat([mae_df, temp_mae_df])

            temp_rmse_df = rmse_df.query('P_0_1==0.0 and iteration == 0').iloc[0:1]
            temp_rmse_df['iteration'] = iter_                
            rmse_df = pd.concat([rmse_df, temp_rmse_df])
            
            run_counter += 1
                
            continue
        
        # Initialize dataframe to store results
        temp_rmse_df = pd.DataFrame()
        temp_rmse_df['P_0_1'] = [prob_0_1]
        temp_rmse_df['P_1_0'] = [prob_1_0]
        temp_rmse_df['num_series'] = [n_miss_series]
        temp_rmse_df['iteration'] = [iter_]

        # Dataframe to store predictions
        temp_Predictions = pd.DataFrame(data = [], columns = models)
            
        # generate missing data
        miss_ind = np.zeros((len(testPred), len(plant_ids)))
                            
        Prob_matrix = np.array([[1-prob_0_1, prob_0_1], [prob_1_0, 1-prob_1_0]])
        
        # Sample missing series 
        if n_miss_series > 1:
            series_missing = np.random.choice(plant_ids, n_miss_series, replace = False)
        elif n_miss_series == 1:
            # pick target series
            series_missing = [target_park]

        for series_name in series_missing:
            # find its column index
            col_index_j = plant_ids.index(series_name)
            
            miss_ind[:,col_index_j] = make_chain(Prob_matrix, 0, len(testPred))
        mask_ind = miss_ind==1
        
        if run_counter%(max(iterations))==0: 
            print(f'Percentage of missing values: {np.round(mask_ind.sum()/mask_ind.size,4)} %')
        
        # Predictors w missing values
        miss_X = scaled_power_df[split:end].copy()[plant_ids]
        miss_X = miss_X[testPred.index[0]:testPred.index[-1]]
        miss_X[mask_ind] = np.nan
        
        miss_X = create_feat_matrix(miss_X, min_lag, max_lag)
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
        
        #### LS model
        lr_pred = projection(lr_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['LR'] = lr_pred.reshape(-1)
                
        #### MLPimp
        mlp_pred = mlp_model.predict(torch.FloatTensor(imp_X.values)).reshape(-1,1)
        temp_Predictions['NN'] = mlp_pred.reshape(-1)
                
        #### Finite Adaptability - Learned Partitions
        
        ### LDR - LS//NN
        
        temp_FA_LEARN_LDR_LR_pred = FA_LEARN_LDR_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions[f'FA-LEARN-LDR-LR-{number_splits}'] = projection(temp_FA_LEARN_LDR_LR_pred).reshape(-1)

        temp_FA_LEARN_LDR_NN_pred = FA_LEARN_LDR_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions[f'FA-LEARN-LDR-NN-{number_splits}'] = projection(temp_FA_LEARN_LDR_NN_pred).reshape(-1)
        
        # ### Static models, no linear decision rules - LS//NN
        
        temp_FA_LEARN_LR_pred = FA_LEARN_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions[f'FA-LEARN-LR-{number_splits}'] = projection(temp_FA_LEARN_LR_pred).reshape(-1)

        temp_FA_LEARN_NN_pred = FA_LEARN_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions[f'FA-LEARN-NN-{number_splits}'] = projection(temp_FA_LEARN_NN_pred).reshape(-1)
        
        temp_error_df = Target.values[max_lag-1:] - temp_Predictions[max_lag-1:]
        
        
        for m in models:
            temp_rmse_df[m] = [rmse(temp_Predictions[m].values[max_lag-1:], Target.values[max_lag-1:])]
        rmse_df = pd.concat([rmse_df, temp_rmse_df])
        run_counter += 1

        # dataframe containing errors
        
        # Diebold-Mariano test for significance
        model_pair = [['LR', 'FA-LEARN-LDR-LR-10'], 
                      ['LR', 'FA-LEARN-LR-10'], 
                      ['FA-LEARN-LDR-LR-10', 'FA-LEARN-LR-10'], 
                      ['LR', 'FA-LEARN-LR-10'], 
                      ['NN', 'FA-LEARN-LDR-NN-10'], 
                      ['FA-LEARN-LDR-LR-10', 'FA-LEARN-LR-10'],
                      ['FA-LEARN-LDR-NN-10', 'FA-LEARN-NN-10']]
        
        for pair in model_pair:
            
            dm_stat, p_val = diebold_mariano_test(temp_error_df[pair[0]], temp_error_df[pair[1]])
            
            # print(f'Pair: {pair}')
            print(f'RMSE:{temp_rmse_df[pair]}')
            print(f'p-val: {p_val}')
            if p_val < 0.01:
                print('Significant at 1%')                
            elif p_val < 0.05:
                print('Significant at 5%')
            elif p_val < 0.1:
                print('Significant at 10%')
            else:
                print('Not significant')
                

        
        
        
        
        
        
        
