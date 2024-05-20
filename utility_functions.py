# -*- coding: utf-8 -*-
"""
Utility functions (probably not used all)

@author: akylas.stratigakos@mines-paristech.fr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import pacf, acf
from scipy.ndimage.interpolation import shift
import itertools
import random
import gurobipy as gp

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


def make_MNAR_chain(t_m, start_term, n, series):
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
        if (series[i]<=.85) and ((series[i]>=0.15)):
            t_m_vary[0] = [.8, .2]
        else:
            t_m_vary[0] = [.999, .001]
        # t_m_vary[0] = [1-series[i]**2, series[i]**2]
        #t_m_vary[0] = [0.9, 0.1]
        chain.append(get_next_term(t_m_vary[chain[-1]]))
    return np.array(chain)

def lagged_predictors_pd(df, col_name, freq, d = 200, thres = .1, intraday = True):
    'Input dataframe, creates lagged predictors of selected column based on PACF'
    PACF = pacf(df[col_name], nlags = d)
    ACF = acf(df[col_name], nlags = d)

    plt.plot(PACF, label = 'PACF')
    plt.plot(ACF, label = 'ACF')
    plt.show()
    
    #Lags = np.argwhere(abs(PACF) > thres) - 1
    Lags = np.where(abs(PACF)>=thres)[0][1:]
    if intraday == False:
        Lags = Lags[Lags> (int(freq*24)-1) ]
    name = col_name+'_'
    name_list = []
    for lag in Lags:
        temp_name = name+str(int(1//freq)*lag)
        df[temp_name] = shift(df[col_name], lag)
        name_list.append(temp_name)
    return df, name_list

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
    return np.sqrt(np.square(np.array(predictions) - actual).mean(0))

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
    
def VaR(data, quant = .05, digits = 3):
    ''' Evaluates Value at Risk at quant-level'''
    if digits is None:
        return np.quantile(data, q = quant)
    else:
        return round(np.quantile(data, q = quant), digits)

def CVaR(data, quant = .05, digits = 3):
    ''' Evaluates Conditional Value at Risk at quant-level'''

    VaR = np.quantile(data, q = quant)
    if digits is None:
        return data[data<=VaR].mean()
    else:
        return round(data[data<=VaR].mean(), digits)


def newsvendor_cvar(scenarios, weights, quant, e = 0.05, risk_aversion = 0.5):

    ''' Weights SAA:
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    
    if scenarios.ndim>1:
        target_scen = scenarios.copy().reshape(-1)
    else:
        target_scen = scenarios.copy()

    n_scen = len(target_scen)
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # target variable
    offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
    loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
                    
    ### CVaR variables (follows Georghiou, Kuhn, et al.)
    beta = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name='VaR')
    zeta = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0)  # Aux
    cvar = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    #m.addConstrs(loss[i] >= quant*(scenarios[i] - offer) for i in range(n_scen))
    #m.addConstrs(loss[i] >= (quant-1)*(scenarios[i] - offer) for i in range(n_scen))

    m.addConstr(loss >= quant*(target_scen - offer) )
    m.addConstr(loss >= (quant-1)*(target_scen - offer) )

    # cvar constraints
    m.addConstr( zeta >=  -beta + loss )
    m.addConstr( cvar == beta + (1/e)*(zeta@weights))
    
    m.setObjective( (1-risk_aversion)*(weights@loss) + risk_aversion*cvar, gp.GRB.MINIMIZE)
    m.optimize()
    
    #plt.plot(loss.X)
    #plt.plot((quant*(scenarios-offer.X)))
    #plt.plot(((quant-1)*(scenarios-offer.X)))
    #plt.show()
    
    #print(scenarios)    
    
    return offer.X

def reg_trading_opt(scenarios, weights, quant, risk_aversion = 0.5):

    ''' Weights SAA:
        scenarios: support/ fixed locations
        weights: the learned probabilities'''
    
    if scenarios.ndim>1:
        target_scen = scenarios.copy().reshape(-1)
    else:
        target_scen = scenarios.copy()

    n_scen = len(target_scen)
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # target variable
    offer = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'offer')
    deviation = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'offer')
    loss = m.addMVar(n_scen, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')

    m.addConstr(deviation == (target_scen - offer) )
    
    m.addConstr(loss >= quant*(target_scen - offer) )
    m.addConstr(loss >= (quant-1)*(target_scen - offer) )
    
    m.setObjective( (1-risk_aversion)*(weights@loss) + risk_aversion*(deviation*deviation)@weights, gp.GRB.MINIMIZE)
    m.optimize()
        
    return offer.X