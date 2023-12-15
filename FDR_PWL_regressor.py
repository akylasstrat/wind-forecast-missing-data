# -*- coding: utf-8 -*-
"""
Feature deletion robust regression for generic piecewise linear loss

@author: akylas.stratigakos@minesparis.psl.eu
"""

#Import Libraries
import numpy as np
import itertools
import gurobipy as gp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from QR_regressor import *

class FDR_PWL_regressor(object):
  '''Initialize the Feature Deletion Robust Regression w piecewise linear loss
  
  Paremeters:
      [c,b]: vectors that parameterize the piecewise linear loss, c: slope, b/c: break point (check full paper)
      K: number of features that are missing at each sample/ budget of robustness (integer)
              - K = 0: standard regression with piecewise linear loss
              - K = len(target_col): all coefficients set to zero, only fit on remaining features.
      target_col: index of columns that can be deleted
      fix_col: index of columns that can be deleted
      approx: select the type of approximation for the robust counterpart problem
          'reformulation': different features are missing at each sample, pessimistic case. 
                          Interpreration: different features missing at different samples, see [2].
          'affine': affinely adjustable robust counterpart, less pessimistic, see [1].
      fit_lb: lower bound on predictions (ignore)
      
      References:
          [1] Gorissen, B. L., & Den Hertog, D. (2013). Robust counterparts of inequalities containing 
           sums of maxima of linear functions. European Journal of Operational Research, 227(1), 30-43.
          [2] Globerson, Amir, and Sam Roweis. "Nightmare at test time: robust learning by feature deletion." 
          Proceedings of the 23rd international conference on Machine learning. 2006.
      '''
  def __init__(self, K = 2, c = [0.5, -0.5], b = [0,0], feat_cluster = False):
     
    self.num_pwl = len(c)
    self.c = c
    self.b = b
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
        
    if solution == 'reformulation':
        # Different features can be deleted at different samples
        # variables
        fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = fit_lower_bound, name = 'fitted')
        bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
        cost = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'cost')
        d = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')
        loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
            
        # Dual variables
        ell_ = [m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0) for p in range(self.num_pwl)]
        mu_ = [m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY) for p in range(self.num_pwl)]
        t_ = [m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux') for p in range(self.num_pwl)]
                
        # Linear Decision Rules: different set of coefficients for each group
        coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
        fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
    
        # check to avoid degenerate solutions
        if K == len(target_col): 
            m.addConstr( coef == 0 )

        start = time.time()
        
        # Dual Constraints-New version
        for p in range(self.num_pwl):
            m.addConstrs( mu_[p] + ell_[p][:,j] >= np.sign(self.c[p])*X[:,target_col[j]]*coef[j] for j in range(len(target_col)))            
            m.addConstr( t_[p] == K*mu_[p] + ell_[p].sum(1))
    
                
        print('Time to declare: ', time.time()-start)
        m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
                
        print('Solving the problem...')
        for p in range(self.num_pwl):
            m.addConstr( loss*(1/np.abs(self.c[p])) >= np.sign(self.c[p])*(Y.reshape(-1) - fitted + self.b[p]) + t_[p])
    
        # Objective
        m.setObjective((1/n_train_obs)*loss.sum(), gp.GRB.MINIMIZE)                    
        m.optimize()
        
        #coef_fdr = np.append(coef.X, fix_coef.X)
        coef_fdr = np.zeros(total_n_feat)
        for i, col in enumerate(target_col):
            coef_fdr[col] = coef.X[i]
        for i, col in enumerate(fix_col):
            coef_fdr[col] = fix_coef.X[i]

        self.objval = m.ObjVal
        self.coef_ = coef_fdr
        self.bias_ = bias.X
        self.cpu_time = m.Runtime
    
            
    elif solution == 'affine':
        # Same features deleted at all samples/ approximation with affinely adjustable robust counterpart
        
        #### Variables
        
        # Primal
        d = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'epigraph')
        fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
        bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
        coef = m.addMVar((n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
        fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
        
        # Linear decision rules for approximation
        p = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        q = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        ##### Dual variables
        # Sum of absolute values
        z = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        mu = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = 0)

        # Dual variables
        ell_ = [m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0) for p in range(self.num_pwl)]
        mu_ = [m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY) for p in range(self.num_pwl)]
        t_ = [m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux') for p in range(self.num_pwl)]

        start = time.time()

        #### Contraints
        # check to avoid degenerate solutions
        if K == len(target_col): 
            m.addConstr( coef == 0 )

        m.addConstr( d >= p.sum() + mu.sum() + K*z )
        m.addConstr( np.ones((n_feat,1))@z + mu >= sum(q) )
        # Dual Constraints to linearize each sample
        m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
        
        # Dual Constraints/ new version
        for pwl in range(self.num_pwl):            
            m.addConstrs( mu_[pwl] + ell_[pwl][:,j] >= np.sign(self.c[pwl])*X[:,target_col[j]]*coef[j] - (1/np.abs(self.c[pwl]))*q[:,j] for j in range(len(target_col)))                       
            m.addConstr( p*(1/np.abs(self.c[pwl])) >= np.sign(self.c[pwl])*(Y.reshape(-1) - fitted + self.b[pwl]) + K*mu_[pwl] + ell_[pwl].sum(1) )

        # Objective
        m.setObjective((1/n_train_obs)*d.sum(), gp.GRB.MINIMIZE)
        print('Time to declare: ', time.time()-start)

        print('Solving the problem...')        
        m.optimize()
        
        # store output
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
                
    elif solution == 'v-enumeration':

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
            for pwl in range(self.num_pwl):                    
                m.addConstr( loss[:,i] >= self.c[pwl]*(Y.reshape(-1) - 
                                                       (((1-alpha)*X[:,target_col])@coef 
                                                        + X[:,fix_col]@fix_coef 
                                                        + np.ones((n_train_obs,1))@bias) 
                                                       + self.b[pwl]))
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
