# -*- coding: utf-8 -*-
"""
Feature Deletion Robust regression with group of variables

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

class FDR_group_regressor(object):
  '''Initialize the Feature Deletion Robust Regression
  
  Paremeters:
      quant: estimated quantile
      K: number of groups of features that are missing at each sample/ budget of robustness (integer). Special cases:
              - K = 0: standard regression with l1 loss
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
  def __init__(self, K = 2, quant = 0.5):
      
    self.quant = quant
    # For a list of quantiles, declare the mode once and warm-start the solution
    if (type(self.quant) == list) or (type(self.quant) == np.ndarray):
        self.solve_multiple = True
    else:
        self.solve_multiple = False
    self.K = K
            
  def fit(self, X, Y, group_col, fix_col, fit_lb = True, verbose = -1, solution = 'v-enumeration'):
    '''group col: list of lists, each sublist includes the indexes of each group that is deleted at once'''
    
    total_n_feat = X.shape[1]
    n_train_obs = len(Y)
    if fit_lb == True:
        fit_lower_bound = 0
    else:
        fit_lower_bound = -gp.GRB.INFINITY
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    n_feat = sum([len(g) for g in group_col])
    n_groups = len(group_col)
    all_target_cols = [item for sublist in group_col for item in sublist]
    # number of feat per group (need to scale K accordingly)
    f_per_group = len(group_col[0])
    K = self.K
    K_feat = f_per_group*self.K
    # loss quantile and robustness budget
    target_quant = self.quant
    
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    if (Y==0).all():
        print('Y = 0: skip training')
        self.coef_ = 0
        self.bias_ = 0
        return

    print('Setting up GUROBI model...')

    if K == 0:

        if self.solve_multiple == False:
            fdr_model = QR_regressor(quantile = target_quant)    
            fdr_model.fit(X, Y)
            
            self.coef_ = fdr_model.coef_
            self.bias_ = fdr_model.bias_
            self.cpu_time = fdr_model.cpu_time
        
        elif self.solve_multiple == True:
        
                # Solve for mutliple quantiles w warm start
                self.coef_ = []
                self.bias_ = []        
                for q in self.quant:
                    print('Quantile: ',q)

                    fdr_model = QR_regressor(quantile = q)    
                    fdr_model.fit(X, Y)
                    
                    self.coef_.append(fdr_model.coef_)
                    self.bias_.append(fdr_model.bias_)    

        return
    elif K>0:
        if solution == 'reformulation':
            # Create incidence matrix 
            M = []
            for group in group_col:
                for i in range(len(group[:-1])):
                    temp = np.zeros(n_feat)
                    temp[group[i]] = 1
                    temp[group[i+1]] = -1        
                    M.append(temp)
            M = np.array(M)
            # Different features can be deleted at different samples
            # variables
            fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = fit_lower_bound, name = 'fitted')
            bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
            cost = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'cost')
            d = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')
            loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
                
            # Dual variables
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
            pi_up = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(X.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
            pi_down = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
                
            # Linear Decision Rules: different set of coefficients for each group
            coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
        
            # check to avoid degenerate solutions
            if self.K == n_groups: 
                m.addConstr( coef == 0 )

            start = time.time()
            # Dual Constraints
            m.addConstrs( np.ones((n_feat,1))@mu_up[i] + ell_up[i] + M.T@pi_up[i] >= sp.diags(X[i,all_target_cols])@coef for i in range(n_train_obs))
            m.addConstrs( t_up[i] == K_feat*mu_up[i] + ell_up[i].sum() for i in range(n_train_obs))
        
            m.addConstrs( np.ones((n_feat,1))@mu_down[i] + ell_down[i] + M.T@pi_down[i] >= -sp.diags(X[i, all_target_cols])@coef for i in range(n_train_obs))
            m.addConstrs( t_down[i] == K_feat*mu_down[i] + ell_down[i].sum() for i in range(n_train_obs))    
        
            '''
            for i in range(len(X)):
                if i%2500==0: print(i)
                m.addConstr( np.ones((n_feat,1))@mu_up[i] + ell_up[i] + M.T@pi_up[i] >= sp.diags(X[i,all_target_cols])@coef )
                m.addConstr( t_up[i] == K*mu_up[i] + ell_up[i].sum() )
            
                m.addConstr( np.ones((n_feat,1))@mu_down[i] + ell_down[i] + M.T@pi_down[i] >= -sp.diags(X[i, all_target_cols])@coef )
                m.addConstr( t_down[i] == K*mu_down[i] + ell_down[i].sum() )    
            
            '''
            
            print('Time to declare: ', time.time()-start)
            m.addConstr( fitted == X[:,all_target_cols]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            #m.addConstrs( loss[i] >= d[i]@d[i] for i in range(n_train_obs))
            
            print('Solving the problem...')
        
            if self.solve_multiple == False:
            
                m.addConstr( loss >= self.quant*(Y.reshape(-1) - fitted + t_up))
                m.addConstr( loss >= (1-self.quant)*(-Y.reshape(-1) + fitted + t_down))
            
                # Objective
                m.setObjective((1/n_train_obs)*loss.sum(), gp.GRB.MINIMIZE)                    
                m.optimize()
                coef_fdr = np.append(coef.X, fix_coef.X)
    
                self.objval = m.ObjVal
                self.coef_ = coef_fdr
                self.bias_ = bias.X
                self.cpu_time = m.Runtime
            
                return 
            elif self.solve_multiple == True:
        
                # Solve for mutliple quantiles w warm start
                self.coef_ = []
                self.bias_ = []        
                for q in self.quant:
                    
                    c1 = m.addConstr( loss >= q*(Y.reshape(-1) - fitted + t_up))
                    c2 = m.addConstr( loss >= (1-q)*(-Y.reshape(-1) + fitted + t_down))
                    # Objective
                    m.setObjective((1/n_train_obs)*loss.sum(), gp.GRB.MINIMIZE)
                    m.optimize()
                    
                    # Remove constraints
                    for constraint in [c1,c2]: m.remove(constraint)
                    
                    coef_fdr = np.append(coef.X, fix_coef.X)
                    self.coef_.append(coef_fdr)
                    self.bias_.append(bias.X)    
                return
            
        elif solution == 'affine':
            # Same features deleted at all samples/ approximation with affinely adjustable robust counterpart
            # Create incidence matrix 
            M = []
            for group in group_col:
                for i in range(len(group[:-1])):
                    temp = np.zeros(n_feat)
                    temp[group[i]] = 1
                    temp[group[i+1]] = -1        
                    M.append(temp)
            M = np.array(M)

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
            pi = m.addMVar(M.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            # positive part of absolute value per sample
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            pi_up = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            # negative part of absolute value per sample
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            pi_down = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            start = time.time()
    
            #### Contraints
            # check to avoid degenerate solutions
            if self.K == n_groups: 
                m.addConstr( coef == 0 )
    
            m.addConstr( d >= p.sum() + mu.sum() + K_feat*z )
            m.addConstr( np.ones((n_feat,1))@z + mu + M.T@pi >= sum(q) )
    
            # Dual Constraints to linearize each sample
            m.addConstr( fitted == X[:,all_target_cols]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            # check if a single quantile is required                                                
            if self.solve_multiple == False:
                            
                m.addConstrs( np.ones((n_feat,1))@mu_up[i] + ell_up[i] + M.T@pi_up[i] >= sp.diags(X[i,all_target_cols])@coef - (1/self.quant)*q[i] for i in range(n_train_obs))
                m.addConstrs( np.ones((n_feat,1))@mu_down[i] + ell_down[i] + M.T@pi_down[i] >= -sp.diags(X[i,all_target_cols])@coef - (1/(1-self.quant))*q[i] for i in range(n_train_obs))
                m.addConstrs( p[i] >= self.quant*(Y[i] - fitted[i] + K_feat*mu_up[i] + ell_up[i].sum()) for i in range(n_train_obs))
                m.addConstrs( p[i] >= (1-self.quant)*(-Y[i] + fitted[i] + K_feat*mu_down[i] + ell_down[i].sum()) for i in range(n_train_obs))
        
                # Objective
                m.setObjective((1/n_train_obs)*d.sum(), gp.GRB.MINIMIZE)
                print('Time to declare: ', time.time()-start)

                print('Solving the problem...')        
                m.optimize()
                
                # store output
                self.objval = m.ObjVal
                coef_fdr = np.append(coef.X, fix_coef.X)
                self.coef_ = coef_fdr
                self.bias_ = bias.X
                self.cpu_time = m.Runtime
                
            elif self.solve_multiple == True:
        
                # Solve for mutliple quantiles w warm start
                self.coef_ = []
                self.bias_ = []        
                
                for qnt in self.quant:
                    print('Quantile: ',qnt)
                    c1 = m.addConstrs( np.ones((n_feat,1))@mu_up[i] + ell_up[i] + M.T@pi_up[i]>= sp.diags(X[i,all_target_cols])@coef - (1/qnt)*q[i] for i in range(n_train_obs))
                    c2 = m.addConstrs( np.ones((n_feat,1))@mu_down[i] + ell_down[i] + M.T@pi_down[i]>= -sp.diags(X[i,all_target_cols])@coef - (1/(1-qnt))*q[i] for i in range(n_train_obs))
                    c3 = m.addConstrs( p[i] >= qnt*(Y[i] - fitted[i] + K_feat*mu_up[i] + ell_up[i].sum()) for i in range(n_train_obs))
                    c4 = m.addConstrs( p[i] >= (1-qnt)*(-Y[i] + fitted[i] + K_feat*mu_down[i] + ell_down[i].sum()) for i in range(n_train_obs))
            
                    # Objective
                    m.setObjective((1/n_train_obs)*d.sum(), gp.GRB.MINIMIZE)
                    #print('Time to declare: ', time.time()-start)
    
                    print('Solving the problem...')        
                    m.optimize()
                    
                    # Remove constraints
                    for constraint in [c1,c2,c3,c4]: m.remove(constraint)
                    
                    coef_fdr = np.append(coef.X, fix_coef.X)
                    self.coef_.append(coef_fdr)
                    self.bias_.append(bias.X)    
                return

        elif solution == 'v-enumeration':
            
            # Set of vertices (combinations of groups)
            V = list(itertools.combinations(range(len(group_col)), K))
            print('Number of vertices: ', len(V))
            #V = [item for sublist in [list(itertools.combinations(range(X.shape[1]), k)) for k in range(1,K+1)] for item in sublist]
            
            if self.solve_multiple == False:
                
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
                    # for each group combination, find specific indexes
                    alpha = np.zeros((n_train_obs, n_feat))
                    for j in v:
                        # set all group features to missing
                        group_ind = group_col[j]
                        alpha[:,group_ind] = 1

                    # constraint for vertex
                    m.addConstr( loss[:,i] >= self.quant*(Y.reshape(-1) - (((1-alpha)*X[:,all_target_cols])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
                    m.addConstr( loss[:,i] >= (1-self.quant)*(-Y.reshape(-1) + (((1-alpha)*X[:,all_target_cols])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
                    m.addConstr( xi >= loss[:,i].sum()/n_train_obs)
                    
                m.setObjective(xi.sum(), gp.GRB.MINIMIZE)                
                m.optimize()
                
                self.objval = m.ObjVal
                coef_fdr = np.append(coef.X, fix_coef.X)
                self.coef_ = coef_fdr
                self.bias_ = bias.X
                self.cpu_time = m.Runtime
            elif self.solve_multiple:
                
                self.coef_ = []
                self.bias_ = []     

                m = gp.Model()
                m.setParam('OutputFlag', 0)
        
                # Variables
                fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
                bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
                xi = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux')
                coef = m.addMVar((n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
                fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
                
                loss = m.addMVar((n_train_obs, len(V)), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
        
                for qnt in self.quant:
                    for i, v in enumerate(V):
                        # for each group combination, find specific indexes
                        alpha = np.zeros((n_train_obs, n_feat))
                        for j in v:
                            group_ind = group_col[j]
                            alpha[:,group_ind] = 1
    
                        # constraint for vertex
                        c1 = m.addConstr( loss[:,i] >= qnt*(Y.reshape(-1) - (((1-alpha)*X[:,all_target_cols])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
                        c2 = m.addConstr( loss[:,i] >= (1-qnt)*(-Y.reshape(-1) + (((1-alpha)*X[:,all_target_cols])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
                        c3 = m.addConstr( xi >= loss[:,i].sum()/n_train_obs)
                        
                    m.setObjective(xi.sum(), gp.GRB.MINIMIZE)                
                    m.optimize()

                    for constraint in [c1,c2,c3]: m.remove(constraint)
                    
                    coef_fdr = np.append(coef.X, fix_coef.X)
                    self.coef_.append(coef_fdr)
                    self.bias_.append(bias.X)
                
        return

        
        
  def predict(self, X):
    if self.solve_multiple == False:
        predictions = X@self.coef_ + self.bias_
        return np.array(predictions)

    elif self.solve_multiple == True:
        # Return list of np arrays for each quantile
        predictions = []
        for i, q in enumerate(self.quant):            
            predictions.append(np.array(X@self.coef_[i] + self.bias_[i]))
        return np.array(predictions).T
  

  def fit_admm(self, X, Y, target_col, fix_col, n_blocks = 4,
               rho = 1, e_primal = 1e-4, e_dual = 1e-2, 
                     max_iter = 1000, verbose = -1):

    if self.solve_multiple:
        print('Warning: ADMM implemented for a single model only')

    A = X.copy()
    b = Y.reshape(-1).copy()
    n_samples = len(A)
    K = self.K
    splits = np.arange(0, n_samples+1, n_samples//n_blocks)

    self.admm_ = {}

    n_feat = X.shape[1]
    primal_res = [np.zeros((n_blocks, n_feat))]
    dual_res = [0]
    n_obs_block = n_samples//n_blocks
    
    # Initialize gurobi model (only works for same sample length!!)
    #!!!!! This is for the x-updates: probably I can find an analytical solution with soft-thresholding
    m = gp.Model()
    
    target_quant = self.quant
    
    sub_prob = [] 
    # Initialize all gurobi models
    for j in range(n_blocks):
        X = A[splits[j]:splits[j+1]]
        Y = b[splits[j]:splits[j+1]]
        fdr = FDR_regressor(K = self.K, quant = self.quant)
        temp_model = fdr.gurobi_model(X, Y, target_col, fix_col, verbose = verbose)
        sub_prob.append(temp_model)
    

    n_train_obs = len(Y)    
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    
    # Store the iterations (!!! bias must be included)
    coef_t = [np.zeros((n_blocks, n_feat+1))]
    coef_ave = [coef_t[-1].mean(axis=0)]
    lambda_ = [np.zeros((n_blocks, n_feat+1))] # dual variable for each sample

    primal_res = [np.zeros((n_blocks, n_feat+1))]
    dual_res = [np.zeros(coef_ave[0].shape)]

    for i in range(max_iter):
        coef_t.append(np.zeros((n_blocks, n_feat+1)))
        if i%10==0: print('iteration: ',i)

        # Loop over blocks    
        for j in range(n_blocks):
            X = A[splits[j]:splits[j+1]]
            Y = b[splits[j]:splits[j+1]]
        
            # Solve (analytical solution ??)
            c2 = sub_prob[j].addConstr( sub_prob[j]._vars['aux']\
                                           == sub_prob[j]._vars['coef_all']-coef_ave[-1])
    
            c3 = sub_prob[j].addConstr( sub_prob[j]._vars['reg'] == \
                             lambda_[-1][j]@sub_prob[j]._vars['aux'] +rho/2*(sub_prob[j]._vars['aux']@sub_prob[j]._vars['aux']))
    
            sub_prob[j].optimize()
            # store new values
            coef_t[-1][j] = sub_prob[j]._vars['coef_all'].X
        
            for constraint in [c2,c3]: 
                sub_prob[j].remove(constraint)
            #m.reset()
    
        # x average update
        coef_ave.append(coef_t[-1].mean(axis=0))

        # dual update
        new_lambda_ = lambda_[-1] + rho*(coef_t[-1] - coef_ave[-1])
        lambda_.append(new_lambda_)
    
        primal_res.append(coef_t[-1] - coef_ave[-1])
        dual_res.append(-rho*(coef_ave[-1] - coef_ave[-2]))
        # check convergence
        if (np.linalg.norm(primal_res[-1])<=e_primal) and (n_blocks*np.linalg.norm(dual_res[-1]) <= e_dual):
            print('Convergence')
            break
        if i%25==0:
            plt.plot([np.linalg.norm(res) for res in primal_res[1:]], label = 'Primal residual')
            plt.plot(np.linalg.norm(np.array(dual_res),axis=1)[1:], label = 'Dual residual')
            plt.legend()
            plt.show()
            
    self.admm_['coef_t'] = coef_t
    self.admm_['coef_ave'] = coef_ave
    self.admm_['primal_res'] = primal_res
    self.admm_['dual_res'] = dual_res

    self.coef_ = coef_ave[-1][1:]
    self.bias_ = coef_ave[-1][0]
    self.cpu_time = m.Runtime

    return

  def gurobi_model(self, X, Y, target_col, fix_col, verbose = -1):
    total_n_feat = X.shape[1]
    n_train_obs = len(Y)    
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    n_feat = len(target_col)

    # loss quantile and robustness budget
    target_quant = self.quant
    K = self.K
    
    m = gp.Model()
    m._vars = {}    

    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    print('Setting up GUROBI model...')
    
    # Variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'fitted')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
    #cost = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'cost')
    loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    
    # Dual variables
    ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
    mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    t_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')

    ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
    mu_down = m.addMVar(X.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    t_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
        
    # Linear Decision Rules: different set of coefficients for each group
    coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
    # Extra parameters for ADMM
    coef_all = m.addMVar(total_n_feat+1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    aux = m.addMVar(total_n_feat+1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')
    reg = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')

    m._vars['coef_all'] = coef_all 
    m._vars['aux'] = aux 
    m._vars['reg'] = reg 

    # Dual Constraints
    m.addConstr( coef_all[0] == bias)    
    m.addConstr( coef_all[1:coef.shape[0]+1] == coef)
    m.addConstr( coef_all[coef.shape[0]+1:] == fix_coef)

    m.addConstrs( np.ones((n_feat,1))@mu_up[i] + ell_up[i] >= sp.diags(X[i,target_col])@coef for i in range(n_train_obs))
    m.addConstrs( t_up[i] == K*mu_up[i] + ell_up[i].sum() for i in range(n_train_obs))

    m.addConstrs( np.ones((n_feat,1))@mu_down[i] + ell_down[i] >= -sp.diags(X[i, target_col])@coef for i in range(n_train_obs))
    m.addConstrs( t_down[i] == K*mu_down[i] + ell_down[i].sum() for i in range(n_train_obs))    

    m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
    
    #m.addConstrs( loss[i] >= d[i]@d[i] for i in range(n_train_obs))
    #m.addConstrs( np.ones((n_feat,1))@mu_up[i] + ell_up[i] >= -sp.diags(X[i,target_col])@coef for i in range(n_train_obs))

    m.addConstr( loss >= target_quant*(Y.reshape(-1) - fitted + t_up))
    m.addConstr( loss >= (1-target_quant)*(-Y.reshape(-1) + fitted + t_down))
    
    # Objective
    #m.setObjective((1/n_train_obs)*(d@d), gp.GRB.MINIMIZE)
    m.setObjective((1/n_train_obs)*loss.sum() + reg, gp.GRB.MINIMIZE)
        
    return m