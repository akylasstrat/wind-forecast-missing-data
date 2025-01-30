# -*- coding: utf-8 -*-
"""
Testing additional models

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
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.model_selection import GridSearchCV


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
    params['impute'] = True # If True, apply mean imputation for missing features
 
    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2018-01-01' # start of train set
    params['split_date'] = '2018-06-01' # end of train set/start of test set
    params['end_date'] = '2019-01-01'# end of test set
    
    params['min_lag'] = 1
    params['max_lag'] = 2 + params['min_lag']
    # last known measure value, defined lookahed horizon (min_lag == 2, 1-hour ahead predictions with 30min data)

    params['save'] = True # If True, then saves models and results
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv(f'{cd}\\data\\2018_wind_site_5min.csv', index_col = 0, parse_dates=True)
metadata_df = pd.read_csv(f'{cd}\\data\\wind_meta.csv', index_col = 0)

freq = '15min'
target_park = 'Noble Clinton'
horizon = 4
test_MCAR = True
test_MNAR = False
test_Censoring = False
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

all_models = ['Pers', 'LR', 'Lasso', 'Ridge', 'LAD', 'NN'] \
        + ['FA-FIXED-LR', 'FA-FIXED-LDR-LR']  \
        + ['FA-FIXED-NN', 'FA-FIXED-LDR-NN']\
        + [f'FA-LEARN-LDR-LR-{n_splits}' for n_splits in FA_LEARN_LDR_LR_models_dict.keys()] \
        + [f'FA-LEARN-LDR-NN-{n_splits}' for n_splits in FA_LEARN_LDR_NN_models_dict.keys()] \
        + [f'FA-LEARN-LR-{n_splits}' for n_splits in FA_LEARN_LR_models_dict.keys()] \
        + [f'FA-LEARN-NN-{n_splits}' for n_splits in FA_LEARN_NN_models_dict.keys()] \

try:
    mae_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_full.csv', index_col=0)
    rmse_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_full.csv', index_col=0)
    run_counter = len(rmse_df)

except:
    mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'P_0_1', 'P_1_0', 'num_series'])
    rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'P_0_1', 'P_1_0', 'num_series'])
    run_counter = 0

# supress warning
pd.options.mode.chained_assignment = None

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

models_to_add = ['FA-LEARN-LDR-NN-10', 'FA-LEARN-NN-10', 
                 'FA-LEARN-LR-10']

models_to_add_dict = {}

with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LDR_NN_model_10_weather.pickle', 'rb') as handle:
    models_to_add_dict['FA-LEARN-LDR-NN-10'] = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_FA_LEARN_NN_model_10_weather.pickle', 'rb') as handle:
    models_to_add_dict['FA-LEARN-NN-10'] = pickle.load(handle)

with open(f'{trained_models_path}\\{target_park}_FA_LEARN_LR_model_10_weather.pickle', 'rb') as handle:
    models_to_add_dict['FA-LEARN-LR-10'] = pickle.load(handle)
    
for m in models_to_add:
    if m not in rmse_df.columns:
        rmse_df[m] = np.nan
        mae_df[m] = np.nan
        
#%%
# Parameters
# iterations = range(10)
# Probability_0_1 = [.05, .1, .2]
# Probability_1_0 = [.1, .2, 1]
# num_series = [1]

# full_experiment_list = list(itertools.product(iterations, Probability_0_1, Probability_1_0, num_series))
 
for run_counter in range(len(rmse_df)):
    
        
    print(run_counter)
    torch.manual_seed(run_counter)
    
    prob_0_1 = rmse_df['P_0_1'].values[run_counter]
    prob_1_0 = rmse_df['P_1_0'].values[run_counter]
    n_miss_series = rmse_df['num_series'].values[run_counter]
    
    if prob_0_1 == 0:
        continue
    
    if n_miss_series not in [1,8]:
        continue

    if prob_0_1 not in [0.05, 0.1, 0.2]:
        continue

    if prob_1_0 not in [1, 0.1, 0.2]:
        continue
    
    print(prob_0_1)
    print(prob_1_0)
    
    # Generate missing data
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
        #miss_ind[1:,j+1] = miss_ind[:-1,j]
        #miss_ind[1:,j+2] = miss_ind[:-1,j+1]
    mask_ind = miss_ind==1
        
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
    
                
    ##### Iterate over models 
    
    for m in models_to_add:
        temp_model = models_to_add_dict[m]
        
        # Generate predictions
        temp_pred = temp_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_pred = projection(temp_pred).reshape(-1)
        
        # Find rmse and mae
        mae_df[m].iloc[run_counter] = mae(temp_pred[max_lag-1:], Target.values[max_lag-1:])
        rmse_df[m].iloc[run_counter] = rmse(temp_pred[max_lag-1:], Target.values[max_lag-1:])
    
    
        # update .csv file

    if config['save']:
        mae_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_updated.csv')
        rmse_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_updated.csv')
