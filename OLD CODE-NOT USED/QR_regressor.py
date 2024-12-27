# -*- coding: utf-8 -*-
"""
Generic quantile regression

@author: a.stratigakos
"""

#Import Libraries
import numpy as np
import pandas as pd
import gurobipy as gp
import time
import scipy.sparse as sp

class QR_regressor(object):
  '''
  Generic quantile regression using gurobi, resembles the sklearn format.
      '''
  def __init__(self, quantile = 0.5, alpha = 0, fit_intercept = True):
    # define target quantile, penalization term, and whether to include intercept
    self.quantile = quantile
    self.alpha = alpha
    self.fit_intercept = fit_intercept
    
  def fit(self, X, Y, verbose = -1):

    n_train_obs = len(Y)
    n_feat = X.shape[1]

    # target quantile and robustness budget
    target_quant = self.quantile
    
    alpha = self.alpha
    
    # If data are pandas, transform to numpys
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
        X = X.copy().values        

    if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.core.series.Series):
        Y = Y.copy().values        
        
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    print('Setting up GUROBI model...')
    
    ### Problem variables
    # main variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
    coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
    # aux variables
    loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')        
    aux = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    
    ### Constraints
    # fitted values
    m.addConstr( fitted == X@coef + bias)

    # linearize loss for each sample
    m.addConstr( loss >= target_quant*(Y.reshape(-1) - fitted))
    m.addConstr( loss >= (1-target_quant)*(-Y.reshape(-1) + fitted))
    
    # l1 regularization penalty
    m.addConstr( aux >= coef)
    m.addConstr( aux >= -coef)

    ### Objective
    m.setObjective((1/n_train_obs)*loss.sum() + alpha*aux.sum(), gp.GRB.MINIMIZE)
    
    print('Solving the problem...')
    
    m.optimize()
    self.coef_ = coef.X
    self.bias_ = bias.X
    self.cpu_time = m.Runtime
    self.objval = m.ObjVal

    return 
    
  def predict(self, X):
    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)