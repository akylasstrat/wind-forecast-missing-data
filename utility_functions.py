# -*- coding: utf-8 -*-
"""
Utility functions (probably not used all)

@author: akylas.stratigakos@mines-paristech.fr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.ndimage.interpolation import shift
# from statsmodels.tsa.stattools import pacf, acf
import itertools
import random
from sklearn.linear_model import LinearRegression

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


def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

def get_next_term(t_s):
    ''' Helper function for markov chain function'''
    return random.choices([0,1], t_s)[0]

def make_chain(t_m, start_term, n):
    ''' Simulates block missingness with Markov Chains/ transition matrix
        t_m: transition matrix. First row controls the non-missing, second row controls the missing data.
        Row-wise sum must be 1. Example: the average block of missing data has length 10 steps.
        Then set the second row as: t_m[1] = [0.1, 0.9]'''
    if  isinstance(t_m, pd.DataFrame):
        t_m = t_m.copy().values
    chain = [start_term]
    for i in range(n-1):
        chain.append(get_next_term(t_m[chain[-1]]))
    return np.array(chain)


def make_MNAR_chain(t_m, start_term, n, series, pattern):
    ''' Simulates block missingness with Markov Chains/ transition matrix
        Data are not missing at random; the probability depends on the actual value
        t_m: transition matrix. First row controls the non-missing, second row controls the missing data.
        Row-wise sum must be 1. Example: the average block of missing data has length 10 steps.
        Then set the second row as: t_m[1] = [0.1, 0.9]'''
    if  isinstance(t_m, pd.DataFrame):
        t_m = t_m.copy().values
    chain = [start_term]
    for i in range(n-1):
        t_m_vary = t_m.copy()
        # the probability of going missing (first row) depends on the actual value of the series
        # first row varies with the values of the series
        if pattern == 'MNAR':
            if (series[i]<=.95) and (series[i]>=.025):
                t_m_vary[0] = [.99, .01]
            else:
                t_m_vary[0] = [.3, .7]
        elif pattern == 'MNAR_sq':
            t_m_vary[0] = [1-series[i]**2, series[i]**2]
        #t_m_vary[0] = [0.9, 0.1]
        chain.append(get_next_term(t_m_vary[chain[-1]]))
    return np.array(chain)

# def lagged_predictors_pd(df, col_name, freq, d = 200, thres = .1, intraday = True):
#     'Input dataframe, creates lagged predictors of selected column based on PACF'
#     PACF = pacf(df[col_name], nlags = d)
#     ACF = acf(df[col_name], nlags = d)

#     plt.plot(PACF, label = 'PACF')
#     plt.plot(ACF, label = 'ACF')
#     plt.show()
    
#     #Lags = np.argwhere(abs(PACF) > thres) - 1
#     Lags = np.where(abs(PACF)>=thres)[0][1:]
#     if intraday == False:
#         Lags = Lags[Lags> (int(freq*24)-1) ]
#     name = col_name+'_'
#     name_list = []
#     for lag in Lags:
#         temp_name = name+str(int(1//freq)*lag)
#         df[temp_name] = shift(df[col_name], lag)
#         name_list.append(temp_name)
#     return df, name_list

def eval_predictions(pred, target, metric = 'mae'):
    ''' Evaluates determinstic forecasts'''
    if metric == 'mae':
        return np.mean(np.abs(pred-target))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred-target).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred-target)/target)
    elif metric == 'mse':
        return np.square(pred-target).mean()

def mae(predictions, actual):
    ''' Evaluates determinstic forecasts
        Outputs: MAE'''
    return np.abs(np.array(predictions).reshape(-1) - actual.reshape(-1)).mean(0)

def rmse(predictions, actual):
    ''' Evaluates determinstic forecasts
        Outputs: MAE'''
    return np.sqrt(np.square(np.array(predictions).copy().reshape(-1) - actual.copy().reshape(-1)).mean(0))

def eval_point_pred(predictions, actual, digits = None):
    ''' Evaluates determinstic forecasts
        Outputs: MAPE, RMSE, MAE'''
    mape = np.mean(abs( (predictions-actual)/actual) )
    rmse = np.sqrt( np.mean(np.square( predictions-actual) ) )
    mae = np.mean(abs(predictions-actual))
    if digits is None:
        return mape,rmse, mae
    else: 
        return round(mape, digits), round(rmse, digits), round(mae, digits)
    
def pinball(prediction, target, quantiles):
    ''' Evaluates Probabilistic Forecasts, outputs Pinball Loss for specified quantiles'''
    num_quant = len(quantiles)
    pinball_loss = np.maximum( (np.tile(target, (1,num_quant)) - prediction)*quantiles,(prediction - np.tile(target , (1,num_quant) ))*(1-quantiles))
    return pinball_loss  

def CRPS(target, quant_pred, quantiles, digits = None):
    ''' Evaluates Probabilistic Forecasts, outputs CRPS'''
    n = len(quantiles)
    #Conditional prob
    p = 1. * np.arange(n) / (n - 1)
    #Heaviside function
    H = quant_pred > target 
    if digits == None:
        return np.trapz( (H-p)**2, quant_pred).mean()
    else:
        return round(np.trapz( (H-p)**2, quant_pred).mean(), digits)
    
def pit_eval(target, quant_pred, quantiles, plot = False, nbins = 20):
    '''Evaluates Probability Integral Transformation
        returns np.array and plots histogram'''
    #n = len(target)
    #y = np.arange(1, n+1) / n
    y = quantiles
    PIT = [ y[np.where(quant_pred[i,:] >= target[i])[0][0]] if any(quant_pred[i,:] >= target[i]) else 1 for i in range(len(target))]
    PIT = np.asarray(PIT).reshape(len(PIT))
    if plot:
        plt.hist(PIT, bins = nbins)
        plt.show()
    return PIT

def reliability_plot(target, pred, quantiles, boot = 100, label = None):
    ''' Reliability plot with confidence bands'''
    cbands = []
    for j in range(boot):
        #Surgate Observations
        Z = np.random.uniform(0,1,len(pred))
        
        Ind = 1* (Z.reshape(-1,1) < np.tile(quantiles,(len(pred),1)))
        cbands.append(np.mean(Ind, axis = 0))
    
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    cbands = 100*np.sort( np.array(cbands), axis = 0)
    lower = int( .05*boot)
    upper = int( .95*boot)
 
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    plt.vlines(100*quantiles, cbands[lower,:], cbands[upper,:])
    plt.plot(100*quantiles,100*ave_proportion, '-*')
    plt.plot(100*quantiles,100*quantiles, '--')
    plt.legend(['Observed', 'Target'])
    plt.show()
    return

def brier_score(predictions, actual, digits = None):
    ''' Evaluates Brier Score''' 
    if digits == None:
        return np.mean(np.square(predictions-actual))
    else:
        return round(np.mean(np.square(predictions-actual)), digits)
    