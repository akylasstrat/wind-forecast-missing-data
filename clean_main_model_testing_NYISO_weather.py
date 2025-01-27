# -*- coding: utf-8 -*-
"""
Testing robust forecasting models

@author: a.stratigakos@imperial.ac.uk
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
horizon = 24
test_MCAR = False
test_MNAR = False
test_Censoring = True
config['save'] = True
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

models = ['Pers', 'LR', 'Lasso', 'Ridge', 'LAD', 'NN'] \
        + ['FA-FIXED-LR', 'FA-FIXED-LDR-LR']  \
        + ['FA-FIXED-NN', 'FA-FIXED-LDR-NN']\
        + [f'FA-LEARN-LDR-LR-{n_splits}' for n_splits in FA_LEARN_LDR_LR_models_dict.keys()] \
        + [f'FA-LEARN-LDR-NN-{n_splits}' for n_splits in FA_LEARN_LDR_NN_models_dict.keys()] \
        + [f'FA-LEARN-LR-{n_splits}' for n_splits in FA_LEARN_LR_models_dict.keys()] \
        + [f'FA-LEARN-NN-{n_splits}' for n_splits in FA_LEARN_NN_models_dict.keys()] \
            
# try:
#     mae_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_weather.csv', index_col = 0)
#     rmse_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_weather.csv', index_col = 0)
# except:
#     mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
#     rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])

# Parameters
iterations = range(10)
Probability_0_1 = [.01, .05, .1, .2]
Probability_1_0 = [0.4, 0.5, 0.6, 0.7, 0.8]
num_series = [len(plant_ids)]

full_experiment_list = list(itertools.product(Probability_0_1, Probability_1_0, num_series, iterations))

#%%
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
#%%
rmse_per_missing_df = pd.DataFrame()

if test_MCAR:
    print('Test for MCAR mechanism')
    
    for prob_0_1, prob_1_0, n_miss_series, iter_ in full_experiment_list:
        
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
        temp_df = pd.DataFrame()
        temp_df['P_0_1'] = [prob_0_1]
        temp_df['P_1_0'] = [prob_1_0]
        temp_df['num_series'] = [n_miss_series]
        temp_df['iteration'] = [iter_]

        # Dataframe to store predictions
        # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
        temp_Predictions = pd.DataFrame(data = [], columns = models)
            
        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(testPred), len(plant_ids)))
                            
        # if freq in ['15min', '5min']:
        #     P = np.array([[1-perc, perc], [0.1, 0.9]])
        # elif freq in ['30min']:
        #     P = np.array([[1-perc, perc], [0.2, 0.8]])

        Prob_matrix = np.array([[1-prob_0_1, prob_0_1], [prob_1_0, 1-prob_1_0]])
        
        # Sample missing series 
        series_missing = np.random.choice(plant_ids, n_miss_series, replace = False)

        for series_name in series_missing:
            # find its column index
            col_index_j = plant_ids.index(series_name)
            
            miss_ind[:,col_index_j] = make_chain(Prob_matrix, 0, len(testPred))
            #miss_ind[1:,j+1] = miss_ind[:-1,j]
            #miss_ind[1:,j+2] = miss_ind[:-1,j+1]
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
        
        #### Persistence
        pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
        temp_Predictions['Pers'] = pers_pred.reshape(-1)
                
        #### LS model
        lr_pred = projection(lr_model.predict(imp_X).reshape(-1,1))
        temp_Predictions['LR'] = lr_pred.reshape(-1)
                
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
        FA_FIXED_LR_pred = FA_FIXED_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions['FA-FIXED-LR'] = projection(FA_FIXED_LR_pred).reshape(-1)

        ## NN model
        FA_FIXED_NN_pred = FA_FIXED_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions['FA-FIXED-NN'] = projection(FA_FIXED_NN_pred).reshape(-1)

        #### Finite Adaptability - Fixed Partitions - LDR
        ## LS model
        FA_FIXED_LDR_LR_pred = FA_FIXED_LDR_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions['FA-FIXED-LDR-LR'] = projection(FA_FIXED_LDR_LR_pred).reshape(-1)

        ## NN model
        FA_FIXED_LDR_NN_pred = FA_FIXED_LDR_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        temp_Predictions['FA-FIXED-LDR-NN'] = projection(FA_FIXED_LDR_NN_pred).reshape(-1)


        #### Finite Adaptability - Learned Partitions
        
        ### LDR - LS//NN
        
        for number_splits in FA_LEARN_LDR_LR_models_dict.keys():            
            temp_FA_LEARN_LDR_LR_pred = FA_LEARN_LDR_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions[f'FA-LEARN-LDR-LR-{number_splits}'] = projection(temp_FA_LEARN_LDR_LR_pred).reshape(-1)

        for number_splits in FA_LEARN_LDR_NN_models_dict.keys():            
            temp_FA_LEARN_LDR_NN_pred = FA_LEARN_LDR_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions[f'FA-LEARN-LDR-NN-{number_splits}'] = projection(temp_FA_LEARN_LDR_NN_pred).reshape(-1)
        
        # ### Static models, no linear decision rules - LS//NN
        
        for number_splits in FA_LEARN_LR_models_dict.keys():            
            temp_FA_LEARN_LR_pred = FA_LEARN_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions[f'FA-LEARN-LR-{number_splits}'] = projection(temp_FA_LEARN_LR_pred).reshape(-1)

        for number_splits in FA_LEARN_NN_models_dict.keys():            
            temp_FA_LEARN_NN_pred = FA_LEARN_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions[f'FA-LEARN-NN-{number_splits}'] = projection(temp_FA_LEARN_NN_pred).reshape(-1)
        
        temp_error_df = Target.values[max_lag-1:] - temp_Predictions[max_lag-1:]
        
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values[max_lag-1:], Target.values[max_lag-1:])]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values[max_lag-1:], Target.values[max_lag-1:])]
        rmse_df = pd.concat([rmse_df, temp_df])
        run_counter += 1
    
        # RMSE vs number of missing features per each observation
        temp_sq_error = np.square(temp_error_df)
        temp_sq_error['Number missing'] = mask_ind.sum(1)[max_lag-1:]
        
        temp_rmse_per_missing = np.sqrt(temp_sq_error.groupby(['Number missing']).mean()).reset_index()
        
        temp_rmse_per_missing['Count'] = temp_sq_error.groupby(['Number missing']).count().values[:,0]
        
        temp_sq_error.groupby(['Number missing']).mean()       
        temp_sq_error.groupby(['Number missing']).count()['Pers']
        rmse_per_missing_df = pd.concat([rmse_per_missing_df, temp_rmse_per_missing])
        
    
        if config['save']:
            # mae_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_weather.csv')
            # rmse_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_weather.csv')
            # rmse_per_missing_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_vs_missing_features_weather.csv')

            mae_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_full.csv')
            rmse_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_full.csv')
            # rmse_per_missing_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_vs_missing_features_weather.csv')
    
        ls_models = ['LR', 'FA-LEARN-LDR-LR-10', 'FA-FIXED-LDR-LR', 'FA-FIXED-LR', 'FA-LEARN-LR-10']
        # rmse_df.groupby(['P_0_1', 'P_1_0']).mean()[ls_models].plot()
            
        nn_models = ['NN', 'FA-LEARN-LDR-NN-10', 'FA-FIXED-LDR-NN', 'FA-FIXED-NN', 'FA-LEARN-NN-10']
        # rmse_df.groupby(['P_0_1', 'P_1_0']).mean()[nn_models].plot()
        
      
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


if test_MNAR:
    print('Test for MNAR mechanism')
    for iter_ in range(iterations):
        print(f'Iteration:{iter_}')
        torch.manual_seed(run_counter)
        # Dataframe to store predictions
        # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
        temp_Predictions = pd.DataFrame(data = [], columns = models)
    
        # Initialize dataframe to store results
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [0]
        temp_df['iteration'] = [iter_]
        
        # generate missing data
        miss_ind = np.zeros((len(testPred), len(plant_ids)))
        
        if freq in ['15min', '5min']:
            P_init = np.array([[.999, .001], [0.1, 0.9]])
            P_norm = np.array([[1-0.01, 0.01], [0.1, 0.9]])
        elif freq in ['30min']:
            P = np.array([[.999, .001], [0.2, 0.8]])
            P_norm = np.array([[1-0.05, 0.05], [0.2, 0.8]])
                
        for j, series in enumerate(series_missing): 
            miss_ind[:,j] = make_MNAR_chain(P_init, 0, len(testPred), scaled_power_df.copy()[series][split:end].values, 'MNAR')
            # if series == target_park:
            #     # Data is MNAR, set values, control the rest within the function 
            #     miss_ind[:,j] = make_MNAR_chain(P, 0, len(testPred), scaled_power_df.copy()[series][split:end].values, 'MNAR')
            # else:
                
            #     miss_ind[:,j] = make_chain(P_norm, 0, len(testPred))
        
        mask_ind = miss_ind==1
        
        if run_counter%iterations==0: print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
            
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
                                
            elif imputation == 'mean':
                for j in range(imp_X.shape[1]):
                    imp_X[np.where(miss_ind[:,j] == 1), j] = mean_imput_values[j]
        
        
            ############ Impute-then-Regress
            
            #### Persistence
            pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
            temp_Predictions['Pers'] = pers_pred.reshape(-1)
                    
            #### LS model
            lr_pred = projection(lr_model.predict(imp_X).reshape(-1,1))
            temp_Predictions['LR'] = lr_pred.reshape(-1)

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
            FA_FIXED_LR_pred = FA_FIXED_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LR'] = projection(FA_FIXED_LR_pred).reshape(-1)
    
            ## NN model
            FA_FIXED_NN_pred = FA_FIXED_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-NN'] = projection(FA_FIXED_NN_pred).reshape(-1)
    
            #### Finite Adaptability - Fixed Partitions - LDR
            ## LS model
            FA_FIXED_LDR_LR_pred = FA_FIXED_LDR_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LDR-LR'] = projection(FA_FIXED_LDR_LR_pred).reshape(-1)
    
            ## NN model
            FA_FIXED_LDR_NN_pred = FA_FIXED_LDR_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LDR-NN'] = projection(FA_FIXED_LDR_NN_pred).reshape(-1)
    
    
            #### Finite Adaptability - Learned Partitions
            
            ### LDR - LS//NN
            
            for number_splits in FA_LEARN_LDR_LR_models_dict.keys():            
                temp_FA_LEARN_LDR_LR_pred = FA_LEARN_LDR_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LDR-LR-{number_splits}'] = projection(temp_FA_LEARN_LDR_LR_pred).reshape(-1)
    
            for number_splits in FA_LEARN_LDR_NN_models_dict.keys():            
                temp_FA_LEARN_LDR_NN_pred = FA_LEARN_LDR_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LDR-NN-{number_splits}'] = projection(temp_FA_LEARN_LDR_NN_pred).reshape(-1)
            
            # ### Static models, no linear decision rules - LS//NN
            
            for number_splits in FA_LEARN_LR_models_dict.keys():            
                temp_FA_LEARN_LR_pred = FA_LEARN_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LR-{number_splits}'] = projection(temp_FA_LEARN_LR_pred).reshape(-1)
    
            for number_splits in FA_LEARN_NN_models_dict.keys():            
                temp_FA_LEARN_NN_pred = FA_LEARN_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-NN-{number_splits}'] = projection(temp_FA_LEARN_NN_pred).reshape(-1)
        
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values, Target.values)]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values, Target.values)]
        rmse_df = pd.concat([rmse_df, temp_df])
        
        run_counter += 1
        
        ls_models = ['LR', 'FA-LEARN-LDR-LR-10', 'FA-LEARN-LR-10', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR']
        rmse_df.groupby(['percentage']).mean()[ls_models].plot(kind='bar')
        
        nn_models = ['NN', 'FA-LEARN-LDR-NN-10', 'FA-LEARN-NN-10', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN']
        rmse_df.groupby(['percentage']).mean()[nn_models].plot(kind='bar')
    
    if config['save']:
        mae_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MNAR_{min_lag}_steps_MAE_results_weather.csv')
        rmse_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MNAR_{min_lag}_steps_RMSE_results_weather.csv')
    
#%%%%%%%%%% CENSORING: Censor data above a specific threshold
#### Note: This falls under MNAR, may replace the current MNAR results (TBD)

mae_df = pd.DataFrame(data = [], columns = models+['upper_bound', 'lower_bound', 'num_series', 'total_missing'])
rmse_df = pd.DataFrame(data = [], columns = models+['upper_bound', 'lower_bound', 'num_series', 'total_missing'])
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

threholds = [ [1, 0.9], [1, 0.8], [1, 0.7], 
                   [0.2, 0], [0.1, 0], [0.05, 0]]

num_series = [1]
series_missing = ['Noble Clinton']

full_experiment_list = list(itertools.product(threholds, num_series))

if test_Censoring:
    print('Test for CENSORING mechanism')    
    
    for temp_threshold, n_series in full_experiment_list:
        ub = temp_threshold[0]
        lb = temp_threshold[1]
        
        print(f'Censor between {lb} and {ub}')
        
        # Dataframe to store predictions
        temp_Predictions = pd.DataFrame(data = [], columns = models)
    
        # Initialize dataframe to store results
        temp_df = pd.DataFrame()
        temp_df['upper_bound'] = [ub]
        temp_df['lower_bound'] = [lb]
        temp_df['num_series'] = [n_series]
        
        # Target series
        measurement_df = scaled_power_df[split:end].copy()[plant_ids]
        miss_ind = np.zeros(measurement_df.shape)
        
        for j, col in enumerate(measurement_df.columns):
            if col in series_missing:
               miss_ind[:,j] =  (measurement_df.copy()[col][split:end].values >= lb) * (measurement_df.copy()[col][split:end].values <= ub)
    
        mask_ind = miss_ind==1
    
        print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
        temp_df['total_missing'] = [100*(mask_ind.sum()/mask_ind.size)]
    
        # Predictors w missing values
        miss_X = measurement_df.copy()
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
                                
            elif imputation == 'mean':
                for j in range(imp_X.shape[1]):
                    imp_X[np.where(miss_ind[:,j] == 1), j] = mean_imput_values[j]
        
    
            #### Persistence
            pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
            temp_Predictions['Pers'] = pers_pred.reshape(-1)
                    
            #### LS model
            lr_pred = projection(lr_model.predict(imp_X).reshape(-1,1))
            temp_Predictions['LR'] = lr_pred.reshape(-1)
    
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
            FA_FIXED_LR_pred = FA_FIXED_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LR'] = projection(FA_FIXED_LR_pred).reshape(-1)
    
            ## NN model
            FA_FIXED_NN_pred = FA_FIXED_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-NN'] = projection(FA_FIXED_NN_pred).reshape(-1)
    
            #### Finite Adaptability - Fixed Partitions - LDR
            ## LS model
            FA_FIXED_LDR_LR_pred = FA_FIXED_LDR_LR_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LDR-LR'] = projection(FA_FIXED_LDR_LR_pred).reshape(-1)
    
            ## NN model
            FA_FIXED_LDR_NN_pred = FA_FIXED_LDR_NN_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
            temp_Predictions['FA-FIXED-LDR-NN'] = projection(FA_FIXED_LDR_NN_pred).reshape(-1)
    
    
            #### Finite Adaptability - Learned Partitions
            
            ### LDR - LS//NN
            
            for number_splits in FA_LEARN_LDR_LR_models_dict.keys():            
                temp_FA_LEARN_LDR_LR_pred = FA_LEARN_LDR_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LDR-LR-{number_splits}'] = projection(temp_FA_LEARN_LDR_LR_pred).reshape(-1)
    
            for number_splits in FA_LEARN_LDR_NN_models_dict.keys():            
                temp_FA_LEARN_LDR_NN_pred = FA_LEARN_LDR_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LDR-NN-{number_splits}'] = projection(temp_FA_LEARN_LDR_NN_pred).reshape(-1)
            
            # ### Static models, no linear decision rules - LS//NN
            
            for number_splits in FA_LEARN_LR_models_dict.keys():            
                temp_FA_LEARN_LR_pred = FA_LEARN_LR_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-LR-{number_splits}'] = projection(temp_FA_LEARN_LR_pred).reshape(-1)
    
            for number_splits in FA_LEARN_NN_models_dict.keys():            
                temp_FA_LEARN_NN_pred = FA_LEARN_NN_models_dict[number_splits].predict(miss_X_zero.values, miss_X.isna().values.astype(int))
                temp_Predictions[f'FA-LEARN-NN-{number_splits}'] = projection(temp_FA_LEARN_NN_pred).reshape(-1)
        
            for m in models:
                temp_df[m] = [mae(temp_Predictions[m].values, Target.values)]
            mae_df = pd.concat([mae_df, temp_df])
            
            for m in models:
                temp_df[m] = [rmse(temp_Predictions[m].values, Target.values)]
            rmse_df = pd.concat([rmse_df, temp_df])
     
        if config['save']:        
            mae_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_CENSOR_{min_lag}_steps_MAE_results_full.csv')
            rmse_df.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_CENSOR_{min_lag}_steps_RMSE_results_full.csv')
    
        ls_models = ['LR', 'FA-LEARN-LDR-LR-5', 'FA-LEARN-LR-10', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR']
        rmse_df.mean()[ls_models].plot(kind='bar')
        plt.show()
    
        nn_models = ['NN', 'FA-LEARN-LDR-NN-10', 'FA-LEARN-NN-10', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN']
        rmse_df.mean()[nn_models].plot(kind='bar')
        plt.show()
