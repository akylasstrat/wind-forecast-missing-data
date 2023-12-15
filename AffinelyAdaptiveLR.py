# -*- coding: utf-8 -*-
"""
Affinely Adaptive Linear Regression, following:
    Bertsimas, Dimitris, Arthur Delarue, and Jean Pauphilet. "Beyond impute-then-regress: Adapting prediction to missing data." arXiv preprint arXiv:2104.03158 (2021).

Missing data are set to 0; estimates base coefficients and linear correction when data are features are missing

@author: a.stratigakos
"""

#Import Libraries
import numpy as np
import itertools
import gurobipy as gp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from QR_regressor import *

class AffinelyAdaptiveLR(object):
    
  '''Initialize the Affinely Adaptive Linear Regression
  
  Paremeters:
      loss: select loss function
      fit_lb: lower bound on predictions (ignore)
      
      References:
          [1] Bertsimas, Dimitris, Arthur Delarue, and Jean Pauphilet. 
          "Beyond impute-then-regress: Adapting prediction to missing data." arXiv preprint arXiv:2104.03158 (2021).
      '''
  def __init__(self, loss = 'mse', fit_intercept = True):
      
    self.fit_intercept = fit_intercept
    self.loss = 'mse'
            
  def fit(self, X, Y, missing_mask, verbose = -1):
    ''' 
        X: features (missing values filled with 0)
        Y: target values
        missing_mask: [0,1] np.array of X.shape, if 1 then entry is missing, else entry is available
        '''

    n_feat = X.shape[1]
    n_train_obs = len(Y)
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)

    print('Setting up GUROBI model...')

    # Different features can be deleted at different samples
    # variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')

    base_coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

    # matrix with Linear Decision Rules that correct base_coef when data are missing
    W_mat = m.addMVar((n_feat, n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    aux = m.addMVar(X.shape, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    residual = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    masked_features = X*(1-missing_mask)

    if self.fit_intercept == False:
        m.addConstr( bias == 0)
    
    m.addConstrs( (fitted[i] ==  masked_features[i]@(base_coef + missing_mask[i]@W_mat) + bias for i in range(n_train_obs)))            

    m.addConstr( residual == Y.reshape(-1) - fitted)            


    if self.loss == 'mse':

        # Objective
        m.setObjective((1/n_train_obs)*(residual@residual), gp.GRB.MINIMIZE)
        
    m.optimize()
        
    # store output
    self.objval = m.ObjVal
    self.base_coef_ = base_coef.X
    self.W_coef_ = W_mat.X
    self.bias_ = bias.X
    self.cpu_time = m.Runtime        
    return
    
            
  def predict(self, X, missing_mask):
    predictions = []
    for i in range(len(X)):
        predictions.append(X[i]@(self.base_coef_ + missing_mask[i]@self.W_coef_) + self.bias_)        
    return np.array(predictions)





