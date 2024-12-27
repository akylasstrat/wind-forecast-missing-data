# -*- coding: utf-8 -*-
"""
Feature-deletion robust regression with quadratic loss/ solved with enumeration
@author: akylas.stratigakos@mineparis.psl.eu
"""

#Import Libraries
import numpy as np
import itertools
import gurobipy as gp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from QR_regressor import *

class FDR_QUAD_regressor(object):
  '''Initialize the Feature Deletion Robust Regression w/ Quadratic loss
  
  Paremeters:
      K: number of features that are missing at each sample/ budget of robustness (integer). Special cases:
              - K = 0: standard regression with piecewise linear loss
              - K = len(target_col): all coefficients set to zero, only fit on remaining features.
      target_col: index of columns that can be deleted
      fix_col: index of columns that can be deleted
      fit_lb: lower bound on predictions (ignore)
      
      References:
          [1] Gorissen, B. L., & Den Hertog, D. (2013). Robust counterparts of inequalities containing 
           sums of maxima of linear functions. European Journal of Operational Research, 227(1), 30-43.
          [2] Globerson, Amir, and Sam Roweis. "Nightmare at test time: robust learning by feature deletion." 
          Proceedings of the 23rd international conference on Machine learning. 2006.
      '''
  def __init__(self, K = 2, feat_cluster = False):
     
    # For a list of quantiles, declare the mode once and warm-start the solution
    self.K = K
            
  def fit(self, X, Y, target_col, fix_col, fit_lb = True, verbose = -1, solution = 'reformulation'):

    total_n_feat = X.shape[1]
    n_train_obs = len(Y)
    if fit_lb == True:
        fit_lower_bound = 0
    else:
        fit_lower_bound = -gp.GRB.INFINITY
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    n_feat = len(target_col)

    # loss quantile and robustness budget
    K = self.K
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    if (Y==0).all():
        print('Y = 0: skip training')
        self.coef_ = np.zeros(X.shape[1])
        self.bias_ = 0
        return

    print('Setting up GUROBI model...')
            
    # Set of vertices
    V = list(itertools.combinations(range(len(target_col)), K))
    print('Number of vertices: ', len(V))
    #V = [item for sublist in [list(itertools.combinations(range(X.shape[1]), k)) for k in range(1,K+1)] for item in sublist]
    
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # Variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
    xi = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
    coef = m.addMVar((n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
    
    loss = m.addMVar((n_train_obs, len(V)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')

    
    for i, v in enumerate(V):
        alpha = np.zeros((n_train_obs, n_feat))
        alpha[:,v] = 1
        m.addConstr( loss[:,i] >= (Y.reshape(-1) - (((1-alpha)*X[:,target_col])@coef 
                                                + X[:,fix_col]@fix_coef 
                                                + np.ones((n_train_obs,1))@bias))*(Y.reshape(-1) - (((1-alpha)*X[:,target_col])@coef 
                                                + X[:,fix_col]@fix_coef 
                                                + np.ones((n_train_obs,1))@bias)))
        m.addConstr( xi >= loss[:,i].sum()/n_train_obs)
        
    m.setObjective(xi.sum(), gp.GRB.MINIMIZE)                
    m.optimize()
    
    self.objval = m.ObjVal
    #coef_fdr = np.append(coef.X, fix_coef.X)
    coef_fdr = np.zeros(total_n_feat)
    for i, col in enumerate(target_col):
        coef_fdr[col] = coef.X[i]
    for i, col in enumerate(fix_col):
        coef_fdr[col] = fix_coef.X[i]

    self.coef_ = coef_fdr
    self.bias_ = bias.X
    self.cpu_time = m.Runtime            
    return
            
  def predict(self, X):
    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)
