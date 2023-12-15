# -*- coding: utf-8 -*-
"""
Checking adaptive linear regression with missing data. See:
    Bertsimas, Dimitris, Arthur Delarue, and Jean Pauphilet. "Beyond impute-then-regress: Adapting prediction to missing data." arXiv preprint arXiv:2104.03158 (2021).

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
from AffinelyAdaptiveLR import *

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

def eval_predictions(pred, target, metric = 'mae'):
    if metric == 'mae':
        return np.mean(np.abs(pred-target))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred-target).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred-target)/target)

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
 
    params['store_folder'] = 'missing-training-test' # folder to save stuff (do not change)
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
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    params['pattern'] = 'random' # per pair of (n_nodes,percentage)
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

if config['save']:
    mae_df.to_csv(f'{cd}\\{case_folder}\\results\\{target_park}_ID_results.csv')


case_folder = config['store_folder']
missing_pattern = config['pattern']


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
target_park = 'p_1257'

# number of lags back to consider
min_lag = config['min_lag']
max_lag = 3
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

#%%%%%%%%% Varying the number of missing observations/ both train and test set

target_pred = Predictors.columns
fixed_pred = []
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = []

n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
error_metric = 'mae'
park_ids = list(power_df.columns.values)

#percentage = [0, .001, .005, .01, .05, .1]
percentage = [0, .001, .005, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['Static', 'ItR-Pers', 'ItR-Mean', 'Adapt', 'Lasso', 'Lasso-opt']
labels = ['$\mathtt{PERS}$', '$\mathtt{LS}$', '$\mathtt{LS}_{\ell_2}$', '$\mathtt{LS}_{\ell_1}$',
           '$\mathtt{LAD}$', '$\mathtt{LAD}_{\ell_1}$','$\mathtt{FDRR-R}$', '$\mathtt{FDRR-AAR}$']


results_output_file_name = f'{cd}\\{case_folder}\\results\\{target_park}_MissingData_{max_lag}_{missing_pattern}.pickle'

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

series_missing = park_ids

config['percentage'] = [0, 0.01, .05, .1, .2]

for perc in config['percentage']:
    if (config['pattern'] == 'MNAR')and(run_counter>1):
        continue
    for iter_ in range(config['iterations']):
        
        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(Y[start:end]), len(series_missing)))
        
        # Generate missing data according to missingness_mechanism
        if config['pattern'] == 'MNAR':
            for j in range(len(series_missing)):                
                # Data is MNAR, set values, control the rest within the function 
                P = np.array([[.999, .001], [0.2, 0.8]])
                miss_ind[:,j] = make_MNAR_chain(P, 0, len(Y[start:end]), power_df.copy()[series_missing].values[:,j].reshape(-1))
        else:
            # Data is MAR/ use the percentage to control the probably of going missing            
            P = np.array([[1-perc, perc], [0.2, 0.8]])

            for j in range(len(series_missing)):
                
                miss_ind[:,j] = make_chain(P, 0, len(Y[start:end]))
        
        miss_ind = miss_ind==1
        #miss_ind_train = miss_ind_train==1
        
        if run_counter%config['iterations']==0: print('Percentage of training observations missing: ', miss_ind.sum()/miss_ind.size)
        
        ###### Predictors w missing values, full data set        
        miss_X = power_df[start:end].copy()[series_missing]
        miss_X[miss_ind] = np.nan
        
        miss_Pred = create_feat_matrix(miss_X, config['min_lag'], config['max_lag'])
        clean_Pred = create_feat_matrix(power_df[start:end].copy(), config['min_lag'], config['max_lag'])
        
        
        # Predictors w missing values, fill with 0
        miss_Pred_zeros = miss_Pred.fillna(0)        
        # Predictors w missing values, imputation w **persistence**
        miss_Pred_pers_imp = miss_Pred.fillna(method = 'ffill').fillna(method = 'bfill')


        # mask variable (1,0) indicating if entry is missing
        train_mask = (miss_Pred[start:split].isna()).astype(int)
        test_mask = (miss_Pred[split:end].isna()).astype(int)

        # mean imputation (when entry is missing, impute with in-sample mean)
        miss_Pred_mean_imp = miss_Pred.copy()
        for j in range(miss_Pred.shape[1]):
            rows_missing_ind = miss_Pred.iloc[:,j].isna().values
            rows_observed_ind = ~miss_Pred.iloc[:,j].isna().values
            
            miss_Pred_mean_imp.iloc[rows_missing_ind,j] = miss_Pred.iloc[rows_observed_ind,j][start:split].mean()
        
        # Create training/test sets/ clean
        trainPred_clean = clean_Pred[start:split]
        testPred_clean = clean_Pred[split:end]

        # Create training/test sets/ fill w 0s
        trainPred_zeros = miss_Pred_zeros[start:split]
        testPred_zeros = miss_Pred_zeros[split:end]
        
        # Create training/test sets/ fill w imputation persistence        
        trainPred_pers = miss_Pred_pers_imp[start:split]
        testPred_pers = miss_Pred_pers_imp[split:end]
        
        trainPred_mean = miss_Pred_mean_imp[start:split]
        testPred_mean = miss_Pred_mean_imp[split:end]        
        
        #%%%%% Train linear models
        
        # Hyperparameter tuning with by cross-validation
        param_grid = {"alpha": [10**pow for pow in range(-5,2)]}

        # Lasso 1: hyperparameter tuning on the clean training data    
        lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 5000), param_grid)
        lasso.fit(trainPred_clean[config['max_lag']-1:], trainY[config['max_lag']-1:])
        
        # Lasso 2: hyperparameter tuning on the noisy training data (post imputation)
        lasso_opt = GridSearchCV(Lasso(fit_intercept = True, max_iter = 5000), param_grid)
        lasso_opt.fit(trainPred_pers[config['max_lag']-1:], trainY[config['max_lag']-1:])
        
        # Static model: all missing data imputed with 0s
        static_lr = LinearRegression(fit_intercept = True)
        static_lr.fit(trainPred_zeros[config['max_lag']-1:], trainY[config['max_lag']-1:])

        # Imp-regress: all missing data imputed with 0s
        imp_lr = LinearRegression(fit_intercept = True)
        imp_lr.fit(trainPred_pers[config['max_lag']-1:], trainY[config['max_lag']-1:])
        
        # impute-regress with mean imputation
        mean_lr = LinearRegression(fit_intercept = True)
        mean_lr.fit(trainPred_mean[config['max_lag']-1:], trainY[config['max_lag']-1:])

        # Affinely adaptive regression
        aalr = AffinelyAdaptiveLR(fit_intercept = True)
        aalr.fit(trainPred_zeros.values, trainY, train_mask.values, verbose = -1)
        
        model_dict = {'Static':static_lr, 'ItR-Pers':imp_lr, 'ItR-Mean':mean_lr, 'Adapt':aalr, 'Lasso':lasso, 'Lasso-opt':lasso_opt}
        #%% Evaluate on test set
        Predictions = pd.DataFrame(data = np.zeros((len(testY), len(models))), columns = models, index = testPred.index)
        
        for m in models:
            if m in ['Static']:
                model_predictions = projection(model_dict[m].predict(testPred_zeros.values).reshape(-1,1))
            elif m in ['ItR-Pers', 'Lasso', 'Lasso-opt']:
                model_predictions = projection(model_dict[m].predict(testPred_pers.values).reshape(-1,1))
            elif m in ['ItR-Mean']:
                model_predictions = projection(model_dict[m].predict(testPred_mean.values).reshape(-1,1))
            elif m in ['Adapt']:
                model_predictions = projection(aalr.predict(testPred_zeros.values, test_mask.values).reshape(-1,1))
                
            if config['scale']: 
                model_predictions = target_scaler.inverse_transform(model_predictions)
            
            Predictions[m] = model_predictions
        
        
        
        '''
        static_pred = projection(static_lr.predict(testPred_zeros).reshape(-1,1))
        imp_pred = projection(imp_lr.predict(testPred_pers).reshape(-1,1))
        mean_pred = projection(mean_lr.predict(testPred_mean).reshape(-1,1))
        aalr_pred = projection(aalr.predict(testPred_zeros.values, test_mask.values).reshape(-1,1))
        
        # both are tested on the imputed test set
        lasso_pred = projection(lasso.predict(testPred_pers).reshape(-1,1))
        lasso_opt_pred = projection(lasso_opt.predict(testPred_pers).reshape(-1,1))

        if config['scale']: 
            static_pred = target_scaler.inverse_transform(static_pred)    
            imp_pred = target_scaler.inverse_transform(imp_pred)    
            mean_pred = target_scaler.inverse_transform(mean_pred)    
            aalr_pred = target_scaler.inverse_transform(aalr_pred)    
            lasso_pred = target_scaler.inverse_transform(lasso_pred)    
            lasso_opt_pred = target_scaler.inverse_transform(lasso_opt_pred)    

        static_mae = eval_predictions(static_pred, Target.values, metric= error_metric)
        imp_mae = eval_predictions(imp_pred, Target.values, metric= error_metric)
        mean_mae = eval_predictions(mean_pred, Target.values, metric= error_metric)
        aalr_mae = eval_predictions(aalr_pred, Target.values, metric= error_metric)
        
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]

        temp_df['Static'] = [static_mae]                        
        temp_df['ItR-Pers'] = [imp_mae]                        
        temp_df['ItR-Mean'] = [mean_mae]                        
        temp_df['Adapt'] = aalr_mae
        '''
        
        temp_df = eval_predictions(Predictions, Target.values, metric= error_metric)
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]
        for m in models: temp_df[m] = eval_predictions(Predictions[m].values.reshape(-1,1), Target.values, metric= error_metric)
        mae_df = pd.concat([mae_df, temp_df])
        run_counter += 1

#%%

mae_df.groupby('percentage').mean()[models].plot(marker='+', ylabel = 'MAE', xticks = config['percentage'], xlabel = '$P_{01}$')
