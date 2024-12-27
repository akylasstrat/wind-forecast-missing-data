# -*- coding: utf-8 -*-
"""
Feature Deletion Robust regression

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

class FDR_regressor(object):
  '''Initialize the Feature Deletion Robust Regression
  
  Paremeters:
      quant: estimated quantile
      K: number of features that are missing at each sample/ budget of robustness (integer). Special cases:
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
  def __init__(self, K = 2, quant = 0.5, feat_cluster = False):
      
    self.quant = quant
    self.feat_cluster = feat_cluster
    # For a list of quantiles, declare the mode once and warm-start the solution
    if (type(self.quant) == list) or (type(self.quant) == np.ndarray):
        self.solve_multiple = True
    else:
        self.solve_multiple = False
    self.K = K
            
  def fit(self, X, Y, target_col, fix_col, fit_lb = False, verbose = -1, solution = 'reformulation'):

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
    target_quant = self.quant
    K = self.K
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    if (Y==0).all():
        print('Y = 0: skip training')
        if self.solve_multiple == False:
            self.coef_ = np.zeros(X.shape[1])
            self.bias_ = 0
        elif self.solve_multiple == True:
            self.coef_ = []
            self.bias_ = []
            for q in self.quant:
                self.coef_.append(np.zeros(X.shape[1]))
                self.bias_.append(0)

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
        
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar((X.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
                
            # Linear Decision Rules: different set of coefficients for each group
            coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
        
            # check to avoid degenerate solutions
            if K == len(target_col): 
                m.addConstr( coef == 0 )

            start = time.time()
            
            # Dual Constraints-New version
            m.addConstrs( mu_up + ell_up[:,j] >= X[:,target_col[j]]*coef[j] for j in range(len(target_col)))            
            m.addConstr( t_up == K*mu_up + ell_up.sum(1))
        
            m.addConstrs( mu_down + ell_down[:,j] >= -X[:,target_col[j]]*coef[j] for j in range(len(target_col)))
            m.addConstr( t_down == K*mu_down + ell_down.sum(1))

            # Dual Constraints-Old version
            #m.addConstrs( np.ones((n_feat))*mu_up[i] + ell_up[i] >= X[i,target_col]*coef for i in range(n_train_obs))            
            #m.addConstrs( t_up[i] == K*mu_up[i] + ell_up[i].sum() for i in range(n_train_obs))

            #m.addConstrs( np.ones((n_feat))*mu_down[i] + ell_down[i] >= -sp.diags(X[i, target_col])@coef for i in range(n_train_obs))
            #m.addConstrs( t_down[i] == K*mu_down[i] + ell_down[i].sum() for i in range(n_train_obs))    
                    
            print('Time to declare: ', time.time()-start)
            m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            #m.addConstrs( loss[i] >= d[i]@d[i] for i in range(n_train_obs))
            
            print('Solving the problem...')

            m.addConstr( loss >= self.quant*(Y.reshape(-1) - fitted + t_up))
            m.addConstr( loss >= (1-self.quant)*(-Y.reshape(-1) + fitted + t_down))
        
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
        
            return 
            
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
    
            # positive part of absolute value per sample
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            # negative part of absolute value per sample
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
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
            m.addConstrs( mu_up + ell_up[:,j] >= X[:,target_col[j]]*coef[j] - (1/self.quant)*q[:,j] for j in range(len(target_col)))            
            m.addConstrs( mu_down + ell_down[:,j] >= -X[:,target_col[j]]*coef[j] - (1/(1-self.quant))*q[:,j] for j in range(len(target_col)))

            m.addConstr( p >= self.quant*(Y.reshape(-1) - fitted + K*mu_up + ell_up.sum(1)) )
            m.addConstr( p >= (1-self.quant)*(-Y.reshape(-1) + fitted + K*mu_down + ell_down.sum(1)) )

            # Dual Constraints/ old version
            #m.addConstrs( np.ones((n_feat))*mu_up[i] + ell_up[i] >= sp.diags(X[i,target_col])@coef - (1/self.quant)*q[i] for i in range(n_train_obs))
            #m.addConstrs( np.ones((n_feat))*mu_down[i] + ell_down[i] >= -sp.diags(X[i,target_col])@coef - (1/(1-self.quant))*q[i] for i in range(n_train_obs))
            #m.addConstrs( p[i] >= self.quant*(Y[i] - fitted[i] + K*mu_up[i] + ell_up[i].sum()) for i in range(n_train_obs))
            #m.addConstrs( p[i] >= (1-self.quant)*(-Y[i] + fitted[i] + K*mu_down[i] + ell_down[i].sum()) for i in range(n_train_obs))
    
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
            
            return
        
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
                                
                m.addConstr( loss[:,i] >= self.quant*(Y.reshape(-1) - (((1-alpha)*X[:,target_col])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
                m.addConstr( loss[:,i] >= (1-self.quant)*(-Y.reshape(-1) + (((1-alpha)*X[:,target_col])@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)))
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
        
        elif solution == 'cutting plane':
            
            # Algorithm Parameters
            coef_t = [np.zeros(X.shape[1])]
            bias_t = [0]
            fitted_t = [X@coef_t[-1]+bias_t[-1]]
            error_t = [np.abs(Y.reshape(-1)-fitted_t[-1])]
            alpha_t = []

            #!!!!!! fix this selected quantile
            LB = [np.abs(Y.reshape(-1)-fitted_t[-1]).mean()]
            UB = []
            epsilon = 1e-3
            # max number of iterations
            iterations = 1000
            M = 1+1e-2

            def soft_limit(model, where):
                ''' Custom termination criteria using function callback
                    -Softlimit: terminates only if the solution is good enough
                    '''
                if where == gp.GRB.Callback.MIP:
                    objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                    objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                    # Check runtime and current gap
                    if objbst > LB[-1] + epsilon:
                        print('Relaxed solution found')
                        model.terminate()
                        
            # setup Lower Bound problem
            lb_m = gp.Model()
            lb_m.setParam('OutputFlag', 0)
            lb_loss = lb_m.addMVar((n_train_obs, iterations), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
            bias = lb_m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
            coef = lb_m.addMVar((n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            xi = lb_m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph-WC loss')


            # setup Upper Bound problem
            upper_m = gp.Model()
            upper_m.setParam('OutputFlag', 0)
            alpha = upper_m.addMVar(X.shape[1], vtype = gp.GRB.BINARY, lb = 0, ub = 1, name = 'alpha')
            gamma = upper_m.addMVar(X.shape[1], vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'gamma')
            aux = upper_m.addMVar(X.shape[1], vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'gamma')

            ub_loss = upper_m.addMVar((n_train_obs), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
            zeta = upper_m.addMVar((n_train_obs), vtype = gp.GRB.BINARY)
            error = upper_m.addMVar((n_train_obs), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            upper_m.addConstr( alpha.sum() == K )
            upper_m.addConstr( ub_loss <= error + M*zeta)
            upper_m.addConstr( ub_loss <= -error + M*(1-zeta))
            upper_m.setObjective(ub_loss.sum()/n_train_obs, gp.GRB.MAXIMIZE)

            # iterative search
            print('Initialize cutting plane algo')
            
            # !!!! Update this also
            # Initial scenario
            starting_v = np.zeros((n_feat))
            starting_v[np.arange(K)] = 1 
            
            alpha_t = [starting_v]
            
            #print(alpha_t[-1].shape)
            #print( ((1 - alpha_t[-1])*X).shape)
            #print(coef.shape)
            #print( (((1 - alpha_t[-1])*X)@coef).shape)
            
            for i in range(iterations):    
                
                if i%5 == 0: print(f'Iteration: {i}')
                
                # Estimate Lower bound
                lb_m.addConstr( lb_loss[:,i] >= Y.reshape(-1) - ( ((1-alpha_t[-1])*X)@coef + bias) )
                lb_m.addConstr( lb_loss[:,i] >= -Y.reshape(-1) + ( ((1 - alpha_t[-1])*X)@coef + bias) )
                
                lb_m.addConstr(xi >= lb_loss[:,i].sum()/n_train_obs)

                lb_m.setObjective(xi.sum(), gp.GRB.MINIMIZE)

                lb_m.optimize()
                
                LB.append(lb_m.ObjVal)
                
                coef_t.append(coef.X)
                bias_t.append(bias.X)
                
                # estimate Upper Bound: worst-case alphas     
                c1 = upper_m.addConstr( aux == np.diag(coef_t[-1])@alpha)
                c2 = upper_m.addConstr( error == Y.reshape(-1) - (X@coef_t[-1] + bias_t[-1]  - X@(coef_t[-1]*alpha) ) )
                upper_m.optimize(soft_limit)
                    
                for c in [c1,c2]: upper_m.remove(c)
                #alpha_t.append(alpha.X)
                alpha_t.append(alpha.X)

                alpha.start = alpha_t[-1]

                UB.append(upper_m.ObjVal)
                    
                # check convergence
                if UB[-1]-LB[-1] <= epsilon:
                    break
                
                plt.plot(LB[1:], label='Lower Bound')
                plt.plot(UB, label='Upper Bound')
                plt.legend()
                plt.show()
            
            #self.objval = m.ObjVal
            self.coef_ = coef_t[-1]
            self.bias_ = bias_t[-1]
            self.cpu_time = m.Runtime            
            
        
        return
            
  def predict(self, X):
    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)
