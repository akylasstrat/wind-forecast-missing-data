# -*- coding: utf-8 -*-
"""
Greedy heuristic algorithm for finite adaptive regression coupled with robust optimization

@author: a.stratigakos
"""

#Import Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from FDR_regressor_test import *
from FDR_regressor import *
from QR_regressor import *

import torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_custom_layers import *

#from decision_solver import *

class Finite_FDRR(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      cost_complexity: Should be included for a simple tree
      spo_weight: Parameter that controls the trade-off between prediction and prescription, to be included
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed

      '''
  def __init__(self, D = 10, Max_models = 5, red_threshold = .01, error_metric = 'mae'):
      
    self.D = D
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    self.error_metric = error_metric
        
  def fit(self, X, Y, target_col, fix_col, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    self.inner_FDR_kwargs = kwargs
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    
    self.children_left_dict = {}
    self.children_right_dict = {}
    self.children_left_dict[0] = -1
    self.children_right_dict[0] = -1
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [np.array(fix_col).copy().astype(int)]

    # features that CAN change missing pattern within current node
    self.target_features = [np.array(target_col).copy().astype(int)]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ### Train Nominal model (no missing data here)

    temp_miss_X = (1-self.missing_pattern[0])*X
    # !!!! Insert pytorch function here
    lr = QR_regressor(fit_intercept=True)
    lr.fit(temp_miss_X, Y)
    #lr = LinearRegression(fit_intercept = True)
    #lr.fit(miss_X, Y)
    
    # Train Adversarially Robust model
    if self.inner_FDR_kwargs['budget'] == 'equality':                
        K_temp = 1
    elif self.inner_FDR_kwargs['budget'] == 'inequality':
        K_temp = len(self.target_features[0])
    
    fdr = FDR_regressor_test(K = K_temp)
    fdr.fit(temp_miss_X, Y, self.target_features[0], self.fixed_features[0], **self.inner_FDR_kwargs)              


    # Nominal and WC loss
    insample_loss = eval_predictions(lr.predict(temp_miss_X), Y, self.error_metric)
    insample_wc_loss = 2*fdr.objval
    
    # Nominal and WC loss
    self.Loss = [insample_loss]
    self.WC_Loss = [insample_wc_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.bias_]
    self.node_model_ = [lr]
    
    self.wc_node_coef_ = [fdr.coef_.reshape(-1)]
    self.wc_node_bias_ = [fdr.bias_]
    self.wc_node_model_ = [fdr]
    
    self.total_models = 1
    
    # keep a list with nodes_IDs that we have not checked yet (are neither parent nodes or set as leaf nodes)
    nodes_ids_candidates = [0]
    self.node_cand_id_ordered = [0]
    temp_node_order = [0]
    
    # for count_, node in enumerate(self.node_cand_id_ordered):
    #     if node not in nodes_ids_candidates: 
    #         continue
    #     if (node != temp_node_order[0]) and (keep_node_aux == False):
    #         continue
    #     elif (node == temp_node_order[0]) or (keep_node_aux == True):
    #         keep_node_aux = True

    for count_, node in enumerate(self.Node_id):

        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models):
            # remove node from list to check (only used for best-first growth)

            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(temp_miss_X), Y, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(temp_miss_X), Y, self.error_metric)

            if (current_node_loss > Best_loss) and (nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)
                # !!!! Not sure we need this
                
                new_lr = QR_regressor(fit_intercept=True)
                new_lr.fit(temp_miss_X, Y)
            
                retrain_loss = eval_predictions(new_lr.predict(temp_miss_X), Y, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:

            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            #self.Node_id.extend([ 2*node + 1, 2*node + 2])
            self.Node_id.extend([ node_id_counter + 1, node_id_counter + 2])
            
            # set depth of new nodes (parent node + 1)
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            left_fdr = FDR_regressor_test(K = K_temp)
            left_fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = 2*left_fdr.objval
            
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            self.Loss_gap.append(self.WC_Loss[-1] - self.Loss[-1]) 
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_coef_.append(self.node_coef_[node])
            self.node_bias_.append(self.node_bias_[node])
            self.node_model_.append(self.node_model_[node])

            self.wc_node_coef_.append(left_fdr.coef_)
            self.wc_node_bias_.append(left_fdr.bias_)
            self.wc_node_model_.append(left_fdr)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X

            right_fdr = FDR_regressor_test(K = K_temp)
            right_fdr.fit(temp_miss_X, Y, right_target_cols, right_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            right_insample_wcloss = 2*right_fdr.objval
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append(self.WC_Loss[-1] - self.Loss[-1]) 

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())

            self.node_coef_.append(new_lr.coef_)
            self.node_bias_.append(new_lr.bias_)
            self.node_model_.append(new_lr)
            
            self.wc_node_coef_.append(right_fdr.coef_)
            self.wc_node_bias_.append(right_fdr.bias_)
            self.wc_node_model_.append(right_fdr)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
                
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2

            else:
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2
                
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)

            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1
        
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0
         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_left_dict[node] != -1)):

             # if missing pattern matches exactly, break loop
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         else:
             # WC model
             Predictions.append( x0@self.wc_node_coef_[node] + self.wc_node_bias_[node] )
     return np.array(Predictions)

class stable_Finite_FDRR(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      cost_complexity: Should be included for a simple tree
      spo_weight: Parameter that controls the trade-off between prediction and prescription, to be included
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed

      '''
  def __init__(self, D = 10, Max_models = 5, red_threshold = .01, error_metric = 'mae'):
      
    self.D = D
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    self.error_metric = error_metric
        
  def fit(self, X, Y, target_col, fix_col, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    self.inner_FDR_kwargs = kwargs
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [np.array(fix_col).copy().astype(int)]

    # features that CAN change missing pattern within current node
    self.target_features = [np.array(target_col).copy().astype(int)]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ### Train Nominal model (no missing data here)

    temp_miss_X = (1-self.missing_pattern[0])*X
    # !!!! Insert pytorch function here
    lr = QR_regressor(fit_intercept=True)
    lr.fit(temp_miss_X, Y)
    #lr = LinearRegression(fit_intercept = True)
    #lr.fit(miss_X, Y)
    
    # Train Adversarially Robust model
    if self.inner_FDR_kwargs['budget'] == 'equality':                
        K_temp = 1
    elif self.inner_FDR_kwargs['budget'] == 'inequality':
        K_temp = len(self.target_features[0])
    
    fdr = FDR_regressor_test(K = K_temp)
    fdr.fit(temp_miss_X, Y, self.target_features[0], self.fixed_features[0], **self.inner_FDR_kwargs)              


    # Nominal and WC loss
    insample_loss = eval_predictions(lr.predict(temp_miss_X), Y, self.error_metric)
    insample_wc_loss = 2*fdr.objval
    
    # Nominal and WC loss
    self.Loss = [insample_loss]
    self.WC_Loss = [insample_wc_loss]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.bias_]
    self.node_model_ = [lr]
    
    self.wc_node_coef_ = [fdr.coef_.reshape(-1)]
    self.wc_node_bias_ = [fdr.bias_]
    self.wc_node_model_ = [fdr]
    
    self.total_models = 1
    
    for node in self.Node_id:
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models):
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(temp_miss_X), Y, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(temp_miss_X), Y, self.error_metric)

            if (current_node_loss > Best_loss) and (nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)
                # !!!! Not sure we need this
                
                new_lr = QR_regressor(fit_intercept=True)
                new_lr.fit(temp_miss_X, Y)
            
                retrain_loss = eval_predictions(new_lr.predict(temp_miss_X), Y, self.error_metric) .reshape(-1,1)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            left_fdr = FDR_regressor_test(K = K_temp)
            left_fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = 2*left_fdr.objval
            
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_coef_.append(self.node_coef_[node])
            self.node_bias_.append(self.node_bias_[node])
            self.node_model_.append(self.node_model_[node])

            self.wc_node_coef_.append(left_fdr.coef_)
            self.wc_node_bias_.append(left_fdr.bias_)
            self.wc_node_model_.append(left_fdr)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X

            right_fdr = FDR_regressor_test(K = K_temp)
            right_fdr.fit(temp_miss_X, Y, right_target_cols, right_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            right_insample_wcloss = 2*right_fdr.objval
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            
            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())

            self.node_coef_.append(new_lr.coef_)
            self.node_bias_.append(new_lr.bias_)
            self.node_model_.append(new_lr)
            
            self.wc_node_coef_.append(right_fdr.coef_)
            self.wc_node_bias_.append(right_fdr.bias_)
            self.wc_node_model_.append(right_fdr)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
                
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
                
            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)

            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1
            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id

  def v2_predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0
         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         else:
             # WC model
             Predictions.append( x0@self.wc_node_coef_[node] + self.wc_node_bias_[node] )
     return np.array(Predictions)
   
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0
         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         else:
             # WC model
             Predictions.append( x0@self.wc_node_coef_[node] + self.wc_node_bias_[node] )
     return np.array(Predictions)


class depth_Finite_FDRR(object):
  '''Adversarially robust regression with finite adaptability.
      Recursively partitions feature space and estimates worst-case model parameters.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Max_models: Maximum number of adaptive models
      red_threshold: threshold in loss improvement to split a new node
      error_metric: Evaluation metric
      max_gap: Maximum percentage gap between optimistic and pessimistic bound
      tree_grow_algo = {'leaf-wise', 'level-wise'}: how to grow the tree. Leaf-wise grows best-first, level-wise moves horizontally and grows each level
              'leaf-wise' is the *suggested* one
      **kwargs: keyword arguments to solve the optimization problem prescribed

      '''
  def __init__(self, D = 10, Max_models = 5, red_threshold = .01, error_metric = 'mae', max_gap = 0.001):
      
    self.D = D
    # Upper bound on loss gap (percentage) across all nodes
    self.max_gap = max_gap
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    self.error_metric = error_metric
        
  def fit(self, X, Y, target_col, fix_col, tree_grow_algo = 'leaf-wise', **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    self.inner_FDR_kwargs = kwargs
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [np.array(fix_col).copy().astype(int)]

    # features that CAN change missing pattern within current node
    self.target_features = [np.array(target_col).copy().astype(int)]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ### Train Nominal model (no missing data here)

    temp_miss_X = (1-self.missing_pattern[0])*X
    # !!!! Insert pytorch function here
    lr = QR_regressor(fit_intercept=True)
    lr.fit(temp_miss_X, Y)
    #lr = LinearRegression(fit_intercept = True)
    #lr.fit(miss_X, Y)
    
    # Train Adversarially Robust model
    if self.inner_FDR_kwargs['budget'] == 'equality':                
        K_temp = 1
    elif self.inner_FDR_kwargs['budget'] == 'inequality':
        K_temp = len(self.target_features[0])
    
    fdr = FDR_regressor_test(K = K_temp)
    fdr.fit(temp_miss_X, Y, self.target_features[0], self.fixed_features[0], **self.inner_FDR_kwargs)              


    # Nominal and WC loss
    insample_loss = eval_predictions(lr.predict(temp_miss_X), Y, self.error_metric)
    insample_wc_loss = 2*fdr.objval
    
    # Nominal and WC loss
    self.Loss = [insample_loss]
    self.WC_Loss = [insample_wc_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.WC_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.bias_]
    self.node_model_ = [lr]
    
    self.wc_node_coef_ = [fdr.coef_.reshape(-1)]
    self.wc_node_bias_ = [fdr.bias_]
    self.wc_node_model_ = [fdr]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes)
    # Lists are updated dynamically at each iteration    
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    
    
    self.list_nodes_candidates_ordered = [0]
        
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(temp_miss_X), Y, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(temp_miss_X), Y, self.error_metric)

            if (current_node_loss > Best_loss) and (nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)
                # !!!! Not sure we need this
                
                new_lr = QR_regressor(fit_intercept=True)
                new_lr.fit(temp_miss_X, Y)
            
                retrain_loss = eval_predictions(new_lr.predict(temp_miss_X), Y, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:

            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            #self.Node_id.extend([ 2*node + 1, 2*node + 2])
            self.Node_id.extend([ node_id_counter + 1, node_id_counter + 2])
            # self.Node_id.extend([ node + 1, node + 2])
                        
            # set depth of new nodes (parent node + 1)
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(left_target_cols)
            print(f'new budget:{K_temp}')
            temp_miss_X = (1-left_missing_pattern)*X
            left_fdr = FDR_regressor_test(K = K_temp)
            left_fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = 2*left_fdr.objval
            
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)            
            self.Loss_gap.append(self.WC_Loss[-1] - self.Loss[-1]) 
            self.Loss_gap_perc.append((self.WC_Loss[-1] - self.Loss[-1])/self.Loss[-1]) 
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_coef_.append(self.node_coef_[node])
            self.node_bias_.append(self.node_bias_[node])
            self.node_model_.append(self.node_model_[node])

            self.wc_node_coef_.append(left_fdr.coef_)
            self.wc_node_bias_.append(left_fdr.bias_)
            self.wc_node_model_.append(left_fdr)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X
            # Copy previous model
            right_fdr = FDR_regressor_test(K = K_temp)
            right_fdr.fit(temp_miss_X, Y, right_target_cols, right_fix_cols, **self.inner_FDR_kwargs)              

            # right_fdr = self.wc_node_model_[node]
            
            # Estimate WC loss and nominal loss
            right_insample_wcloss = 2*right_fdr.objval
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append(self.WC_Loss[-1] - self.Loss[-1]) 
            self.Loss_gap_perc.append((self.WC_Loss[-1] - self.Loss[-1])/self.Loss[-1]) 

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())

            self.node_coef_.append(best_node_coef_)
            self.node_bias_.append(best_node_bias_)
            self.node_model_.append(best_new_model)
            
            self.wc_node_coef_.append(right_fdr.coef_)
            self.wc_node_bias_.append(right_fdr.bias_)
            self.wc_node_model_.append(right_fdr)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
                
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2

            else:
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2
                
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
            
            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
            
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

            # Update the total number nodes
            node_id_counter = node_id_counter + 2

        else:

            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)

            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1
        
            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0
         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_left_dict[node] != -1)):

             # if missing pattern matches exactly, break loop
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         else:
             # WC model
             Predictions.append( x0@self.wc_node_coef_[node] + self.wc_node_bias_[node] )
     return np.array(Predictions)

class newversion_depth_Finite_FDRR(object):
  '''Adversarially robust regression with finite adaptability.
      Recursively partitions feature space and estimates worst-case model parameters.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Max_models: Maximum number of adaptive models
      red_threshold: threshold in loss improvement to split a new node
      error_metric: Evaluation metric
      max_gap: Maximum percentage gap between optimistic and pessimistic bound
      tree_grow_algo = {'leaf-wise', 'level-wise'}: how to grow the tree. Leaf-wise grows best-first, level-wise moves horizontally and grows each level
              'leaf-wise' is the *suggested* one
      **kwargs: keyword arguments to solve the optimization problem prescribed

      '''
  def __init__(self, D = 10, Max_models = 5, red_threshold = .01, error_metric = 'mae', max_gap = 0.001):
      
    self.D = D
    # Upper bound on loss gap (percentage) across all nodes
    self.max_gap = max_gap
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    self.error_metric = error_metric
        
  def fit(self, X, Y, target_col, fix_col, tree_grow_algo = 'leaf-wise', **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    self.inner_FDR_kwargs = kwargs
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [np.array(fix_col).copy().astype(int)]

    # features that CAN change missing pattern within current node
    self.target_features = [np.array(target_col).copy().astype(int)]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ### Train Nominal model (no missing data here)

    temp_miss_X = (1-self.missing_pattern[0])*X
    # !!!! Insert pytorch function here
    lr = QR_regressor(fit_intercept=True)
    lr.fit(temp_miss_X, Y)
    #lr = LinearRegression(fit_intercept = True)
    #lr.fit(miss_X, Y)
    
    # Train Adversarially Robust model
    # if self.inner_FDR_kwargs['budget'] == 'equality':                
    #     K_temp = 1
    # elif self.inner_FDR_kwargs['budget'] == 'inequality':
    #     K_temp = len(self.target_features[0])
    
    
    # Robust-equality 
    eq_fdr = FDR_regressor_test(K = 1)
    eq_fdr.fit(temp_miss_X, Y, self.target_features[0], self.fixed_features[0], budget = 'equality', **self.inner_FDR_kwargs)              

    # Robust-inequality 
    ineq_fdr = FDR_regressor_test(K = len(self.target_features[0]))
    ineq_fdr.fit(temp_miss_X, Y, self.target_features[0], self.fixed_features[0], budget = 'inequality', **self.inner_FDR_kwargs)              

    # Nominal and WC loss
    insample_loss = eval_predictions(lr.predict(temp_miss_X), Y, self.error_metric)
    eq_wc_loss = 2*eq_fdr.objval
    ineq_wc_loss = 2*ineq_fdr.objval
    
    # Nominal and WC loss
    self.Loss = [insample_loss]
    self.eq_wc_Loss = [eq_wc_loss]
    self.ineq_wc_Loss = [ineq_wc_loss]

    self.Loss_gap = [self.ineq_wc_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.ineq_wc_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.bias_]
    self.node_model_ = [lr]
    
    self.eq_wc_node_coef_ = [eq_fdr.coef_.reshape(-1)]
    self.eq_wc_node_bias_ = [eq_fdr.bias_]
    self.eq_wc_node_model_ = [eq_fdr]

    self.ineq_wc_node_coef_ = [ineq_fdr.coef_.reshape(-1)]
    self.ineq_wc_node_bias_ = [ineq_fdr.bias_]
    self.ineq_wc_node_model_ = [ineq_fdr]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes)
    # Lists are updated dynamically at each iteration    
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    
    
    self.list_nodes_candidates_ordered = [0]
        
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(temp_miss_X), Y, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            wc_node_loss = eval_predictions(self.eq_wc_node_model_[node].predict(temp_miss_X), Y, self.error_metric)

            if (current_node_loss > Best_loss) and (nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)
                # !!!! Not sure we need this
                
                new_lr = QR_regressor(fit_intercept=True)
                new_lr.fit(temp_miss_X, Y)
            
                retrain_loss = eval_predictions(new_lr.predict(temp_miss_X), Y, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:

            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            #self.Node_id.extend([ 2*node + 1, 2*node + 2])
            self.Node_id.extend([ node_id_counter + 1, node_id_counter + 2])
            # self.Node_id.extend([ node + 1, node + 2])
                        
            # set depth of new nodes (parent node + 1)
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()
            

            temp_miss_X = (1-left_missing_pattern)*X
            
            # Robust/ equality            
            eq_left_fdr = FDR_regressor_test(K = 1)
            eq_left_fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, budget = 'equality', **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            eq_left_wc_loss = 2*eq_left_fdr.objval

            # Robust/ inequality            
            ineq_left_fdr = FDR_regressor_test(K = len(left_target_cols))
            ineq_left_fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, budget = 'inequality', **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            ineq_left_wc_loss = 2*ineq_left_fdr.objval
            
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.eq_wc_Loss.append(eq_left_wc_loss)            
            self.ineq_wc_Loss.append(ineq_left_wc_loss)            

            self.Loss_gap.append(self.ineq_wc_Loss[-1] - self.Loss[-1]) 
            self.Loss_gap_perc.append((self.ineq_wc_Loss[-1] - self.Loss[-1])/self.Loss[-1]) 
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_coef_.append(self.node_coef_[node])
            self.node_bias_.append(self.node_bias_[node])
            self.node_model_.append(self.node_model_[node])

            self.eq_wc_node_coef_.append(eq_left_fdr.coef_)
            self.eq_wc_node_bias_.append(eq_left_fdr.bias_)
            self.eq_wc_node_model_.append(eq_left_fdr)

            self.ineq_wc_node_coef_.append(ineq_left_fdr.coef_)
            self.ineq_wc_node_bias_.append(ineq_left_fdr.bias_)
            self.ineq_wc_node_model_.append(ineq_left_fdr)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value            
            temp_miss_X = (1-right_missing_pattern)*X

            # Robust equality
            eq_right_fdr = FDR_regressor_test(K = 1)
            eq_right_fdr.fit(temp_miss_X, Y, right_target_cols, right_fix_cols, budget = 'equality', **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            eq_wc_right_wcloss = 2*eq_right_fdr.objval
            

            self.Loss.append(Best_insample_loss)
            self.eq_wc_Loss.append(eq_wc_right_wcloss)            
            self.ineq_wc_Loss.append(self.ineq_wc_Loss[node])            

            self.Loss_gap.append(self.ineq_wc_Loss[-1] - self.Loss[-1]) 
            self.Loss_gap_perc.append((self.ineq_wc_Loss[-1] - self.Loss[-1])/self.Loss[-1]) 

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())

            self.node_coef_.append(best_node_coef_)
            self.node_bias_.append(best_node_bias_)
            self.node_model_.append(best_new_model)
            
            self.eq_wc_node_coef_.append(eq_right_fdr.coef_)
            self.eq_wc_node_bias_.append(eq_right_fdr.bias_)
            self.eq_wc_node_model_.append(eq_right_fdr)

            self.ineq_wc_node_coef_.append(self.ineq_wc_node_coef_[node])
            self.ineq_wc_node_bias_.append(self.ineq_wc_node_bias_[node])
            self.ineq_wc_node_model_.append(self.ineq_wc_node_model_[node])
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
                
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2

            else:
                self.children_left_dict[node] = node_id_counter+1
                self.children_right_dict[node] = node_id_counter+2
                
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
            
            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
            
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

            # Update the total number nodes
            node_id_counter = node_id_counter + 2

        else:

            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)

            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1
        
            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0
         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_left_dict[node] != -1)):

             # if missing pattern matches exactly, break loop
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         elif (m0.sum() == self.total_missing_feat[node] + 1):
             # equality robust model
             Predictions.append( x0@self.eq_wc_node_coef_[node] + self.eq_wc_node_bias_[node] )
         else:
             # inequality robust
             Predictions.append( x0@self.ineq_wc_node_coef_[node] + self.ineq_wc_node_bias_[node] )
     return np.array(Predictions)
 
class FiniteAdaptability_MLP(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed
      '''
  def __init__(self, target_col, fix_col, D = 10, Max_models = 5, red_threshold = .01, error_metric = 'mse', **kwargs):
      
    self.D = D
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.gd_FDRR_params = kwargs


  def fit(self, X, Y, tree_grow_algo = 'leaf-wise', max_gap = 0.001, val_split = 0.05, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''

    self.MLP_train_dict = kwargs
    self.max_gap = max_gap

    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [self.fix_col]

    # features that CAN change missing pattern within current node
    self.target_features = [self.target_col]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ####### Create train/validation data loaders for torch modules
    temp_miss_X = (1-self.missing_pattern[0])*X
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y
        
        train_temp_miss_X = temp_miss_X
        valid_temp_miss_X = temp_miss_X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_miss_X = temp_miss_X[:-n_valid_obs]
        valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
    tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

    train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'])        
    valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'])
        
    ####### Train nominal model
    
    if self.gd_FDRR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_miss_X, trainY)
    else:
        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                                  train_adversarially = False, budget_constraint = self.gd_FDRR_params['budget_constraint'])
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])
        
    ######## Train Adversarially Robust model
    # budget of robustness
    # Nominal loss: is estimated/ WC loss: set to negative value
    if self.gd_FDRR_params['budget_constraint'] == 'equality':                
        gamma_temp = 1
    elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
        gamma_temp = len(self.target_features[0])

    robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                              target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                              train_adversarially = True, 
                              Gamma = gamma_temp, budget_constraint = self.gd_FDRR_params['budget_constraint'])

    optimizer = torch.optim.Adam(robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
    
    # robust_mlp_model.load_state_dict(mlp_model.state_dict(), strict=False)

    robust_mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'], 
                          warm_start = False, attack_type = self.gd_FDRR_params['attack_type'])
    

    # Nominal and WC loss
    insample_loss = eval_predictions(mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
    insample_wc_loss = eval_predictions(robust_mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
        
    print(insample_loss)
    print(insample_wc_loss)
    
    # Nominal, WC loss, and loss gap
    self.Loss = [insample_loss]
    self.WC_Loss = [insample_wc_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.WC_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_model_ = [mlp_model]
    self.wc_node_model_ = [robust_mlp_model]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes), updated dynamically at each iteration
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    

    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'])        
            valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'])
                
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(train_temp_miss_X), trainY, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            
            if (current_node_loss > Best_loss)*(nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)

                ####### Train nominal model with missing data
                
                if self.gd_FDRR_params['hidden_sizes'] == []:
                    # Train Linear Regression model on missing data realization
                    new_mlp_model = LinearRegression(fit_intercept = True)
                    new_mlp_model.fit(train_temp_miss_X, trainY)
                else:
                    # Train Neural Network model on missing data realization
                    new_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                              target_col = self.target_features[node], fix_col = self.fixed_features[node], projection = self.gd_FDRR_params['projection'], 
                                              train_adversarially = False)
    
                    optimizer = torch.optim.Adam(new_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
    
                    new_mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                                          verbose = self.MLP_train_dict['verbose'])

                            
                retrain_loss = eval_predictions(new_mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
                wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(tensor_train_missX), trainY, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                                
                    solution_count = solution_count + 1
                    apply_split = True
                                        
                    # placeholders for node split
                    best_new_model = new_mlp_model
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()

            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            left_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)        
            left_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)

            left_robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = left_target_cols, fix_col = left_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, Gamma = gamma_temp, 
                                      budget_constraint = self.gd_FDRR_params['budget_constraint'])

            optimizer = torch.optim.Adam(left_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            # left_robust_mlp_model.load_state_dict(self.wc_node_model_[node].state_dict(), strict=False)

            left_robust_mlp_model.train_model(left_train_data_loader, left_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'], 
                                  warm_start = False, attack_type = self.gd_FDRR_params['attack_type'])

            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = eval_predictions(left_robust_mlp_model.predict(tensor_train_missX), trainY, self.error_metric)

            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_model_.append(self.node_model_[node])
            self.wc_node_model_.append(left_robust_mlp_model)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)
                
            right_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)        
            right_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)

            right_robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = right_target_cols, fix_col = right_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, 
                                      Gamma = gamma_temp, budget_constraint = self.gd_FDRR_params['budget_constraint'])

            optimizer = torch.optim.Adam(right_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            # right_robust_mlp_model.load_state_dict(self.wc_node_model_[node].state_dict(), strict=False)

            right_robust_mlp_model.train_model(right_train_data_loader, right_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'], 
                                  warm_start = False, attack_type = self.gd_FDRR_params['attack_type'])

            # Estimate WC loss and nominal loss
            right_insample_wcloss = eval_predictions(right_robust_mlp_model.predict(tensor_train_missX), trainY, self.error_metric)
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())


            self.node_model_.append(best_new_model)            
            self.wc_node_model_.append(right_robust_mlp_model)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(right_fix_cols)
            self.target_features.append(right_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)

            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap_perc)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
                        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
                print(list_to_check)
            else:
                list_to_check = list_nodes_candidates

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

                            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( self.node_model_[node].predict(x0).reshape(-1))
         else:
             # WC model
             Predictions.append( self.wc_node_model_[node].predict(x0).reshape(-1))

     return np.array(Predictions).reshape(-1,1)

class FiniteLinear_MLP(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed
      '''
  def __init__(self, target_col, fix_col, D = 10, Max_splits = 5, red_threshold = .01, error_metric = 'mse', **kwargs):
      
    self.D = D
    self.Max_splits = Max_splits
    self.red_threshold = red_threshold
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.gd_FDRR_params = kwargs


  def fit(self, X, Y, tree_grow_algo = 'leaf-wise', max_gap = 0.001, val_split = 0.05, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''

    self.MLP_train_dict = kwargs
    
    # check some training parameters
    if 'weight_decay' not in list(self.MLP_train_dict.keys()):
        self.MLP_train_dict['weight_decay'] = 0
    
    if 'freeze_weights' not in list(self.MLP_train_dict.keys()):
        self.MLP_train_dict['freeze_weights'] = True
        
    self.max_gap = max_gap

    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]

    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [self.fix_col]

    # features that CAN change missing pattern within current node
    self.target_features = [self.target_col]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ####### Create train/validation data loaders for torch modules
    temp_miss_X = (1-self.missing_pattern[0])*X
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y
        
        train_temp_miss_X = temp_miss_X
        valid_temp_miss_X = temp_miss_X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_miss_X = temp_miss_X[:-n_valid_obs]
        valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
    tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

    train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
        
    ####### Train nominal model
    
    if self.gd_FDRR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_miss_X, trainY)
    else:
        torch.manual_seed(0)

        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                                  train_adversarially = False, budget_constraint = self.gd_FDRR_params['budget_constraint'])
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = 1e-2, weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])
        
    ######## Train Adversarially Robust model
    # budget of robustness
    # Nominal loss: is estimated/ WC loss: set to negative value
    if self.gd_FDRR_params['budget_constraint'] == 'equality':                
        gamma_temp = 1
    elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
        gamma_temp = len(self.target_features[0])
        
    robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                              target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                              train_adversarially = True, 
                              Gamma = gamma_temp, budget_constraint = 'inequality')

    optimizer = torch.optim.Adam(robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
    
    # warm start base coeff
    if self.gd_FDRR_params['hidden_sizes'] == []:
        robust_mlp_model.model[0].weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
        robust_mlp_model.model[0].bias.data = torch.FloatTensor(mlp_model.intercept_.reshape(1,-1))
    else:
        robust_mlp_model.load_state_dict(mlp_model.state_dict(), strict=False)

    # robust_mlp_model.sequential_train_model(train_data_loader, valid_data_loader, optimizer, 
    #                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
    #                       attack_type = self.gd_FDRR_params['attack_type'])

    robust_mlp_model.adversarial_train_model(train_data_loader, valid_data_loader, optimizer, 
                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                          attack_type = self.gd_FDRR_params['attack_type'], freeze_weights = self.MLP_train_dict['freeze_weights'])
    

    # Nominal and WC loss
    insample_loss = eval_predictions(mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
    insample_wc_loss = eval_predictions(robust_mlp_model.predict(train_temp_miss_X, self.missing_pattern[0]), trainY, self.error_metric)
        
    print(insample_loss)
    print(robust_mlp_model.best_val_loss)
    
    # Nominal, WC loss, and loss gap
    self.Loss = [insample_loss]
    self.WC_Loss = [robust_mlp_model.best_val_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.WC_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_model_ = [mlp_model]
    self.wc_node_model_ = [robust_mlp_model]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes), updated dynamically at each iteration
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    

    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_splits) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
                
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(train_temp_miss_X), trainY, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            
            if (current_node_loss > Best_loss)*(nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)

                ####### Train nominal model with missing data
                
                if self.gd_FDRR_params['hidden_sizes'] == []:
                    # Train Linear Regression model on missing data realization
                    new_mlp_model = LinearRegression(fit_intercept = True)
                    new_mlp_model.fit(train_temp_miss_X, trainY)
                else:
                    # Train Neural Network model on missing data realization
                    new_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                              target_col = self.target_features[node], fix_col = self.fixed_features[node], projection = self.gd_FDRR_params['projection'], 
                                              train_adversarially = False)
    
                    optimizer = torch.optim.Adam(new_mlp_model.parameters(), lr = 1e-2)
    
                    new_mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                                          verbose = self.MLP_train_dict['verbose'])

                            
                retrain_loss = eval_predictions(new_mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
                wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(tensor_train_missX, temp_missing_pattern), trainY, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                                
                    solution_count = solution_count + 1
                    apply_split = True
                                        
                    # placeholders for node split
                    best_new_model = new_mlp_model
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()

            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            left_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            left_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)

            left_robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = left_target_cols, fix_col = left_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, Gamma = gamma_temp, 
                                      budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                left_robust_mlp_model.model[0].weight.data = torch.FloatTensor(self.node_model_[node].coef_[0].reshape(1,-1))
                left_robust_mlp_model.model[0].bias.data = torch.FloatTensor(self.node_model_[node].intercept_.reshape(1,-1))
            else:
                left_robust_mlp_model.load_state_dict(self.node_model_[node].state_dict(), strict=False)

            optimizer = torch.optim.Adam(left_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            # left_robust_mlp_model.sequential_train_model(left_train_data_loader, left_valid_data_loader, optimizer, 
            #                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
            #                       attack_type = self.gd_FDRR_params['attack_type'])

            left_robust_mlp_model.adversarial_train_model(left_train_data_loader, left_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'], freeze_weights = self.MLP_train_dict['freeze_weights'])
            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = left_robust_mlp_model.best_val_loss

            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_model_.append(self.node_model_[node])
            self.wc_node_model_.append(left_robust_mlp_model)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)
                
            right_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)        
            right_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)

            right_robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = right_target_cols, fix_col = right_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, Gamma = gamma_temp, budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                right_robust_mlp_model.model[0].weight.data = torch.FloatTensor(best_new_model.coef_[0].reshape(1,-1))
                right_robust_mlp_model.model[0].bias.data = torch.FloatTensor(best_new_model.intercept_.reshape(1,-1))
            else:
                right_robust_mlp_model.load_state_dict(best_new_model.state_dict(), strict=False)

            # right_robust_mlp_model.load_state_dict(self.wc_node_model_[node].state_dict(), strict=False)
            right_robust_mlp_model.W.data = self.wc_node_model_[node].W.data
            
            optimizer = torch.optim.Adam(right_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            # right_robust_mlp_model.sequential_train_model(right_train_data_loader, right_valid_data_loader, optimizer, 
            #                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
            #                       attack_type = self.gd_FDRR_params['attack_type'])

            right_robust_mlp_model.adversarial_train_model(right_train_data_loader, right_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'], freeze_weights = self.MLP_train_dict['freeze_weights'])

            # Estimate WC loss and nominal loss
            right_insample_wcloss = right_robust_mlp_model.best_val_loss
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())


            self.node_model_.append(best_new_model)            
            self.wc_node_model_.append(right_robust_mlp_model)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(right_fix_cols)
            self.target_features.append(right_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)

            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap_perc)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
                        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
                print(list_to_check)
            else:
                list_to_check = list_nodes_candidates

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

                            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             # print('New Node: ', node)
             # print(m0)
         
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( self.node_model_[node].predict(x0).reshape(-1))
         else:
             # WC model
             Predictions.append( self.wc_node_model_[node].predict(x0, m0).reshape(-1))

     return np.array(Predictions).reshape(-1,1)

class FiniteAdapt_Greedy(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed
      '''
  def __init__(self, target_col, fix_col, D = 10, Max_splits = 5, red_threshold = .01, error_metric = 'mse', **kwargs):
      
    self.D = D
    self.Max_splits = Max_splits
    self.red_threshold = red_threshold
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.gd_FDRR_params = kwargs


  def fit(self, X, Y, tree_grow_algo = 'leaf-wise', max_gap = 0.001, val_split = 0.05, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''

    self.MLP_train_dict = kwargs
    self.max_gap = max_gap

    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [self.fix_col]

    # features that CAN change missing pattern within current node
    self.target_features = [self.target_col]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ####### Create train/validation data loaders for torch modules
    temp_miss_X = (1-self.missing_pattern[0])*X
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y
        
        train_temp_miss_X = temp_miss_X
        valid_temp_miss_X = temp_miss_X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_miss_X = temp_miss_X[:-n_valid_obs]
        valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
    tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

    train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
        
    ####### Train nominal model
    
    if self.gd_FDRR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_miss_X, trainY)
    else:
        torch.manual_seed(0)

        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                                  train_adversarially = False, budget_constraint = self.gd_FDRR_params['budget_constraint'])
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])
        
    ######## Train Adversarially Robust model
    # budget of robustness
    # Nominal loss: is estimated/ WC loss: set to negative value
    if self.gd_FDRR_params['budget_constraint'] == 'equality':                
        gamma_temp = 1
    elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
        gamma_temp = len(self.target_features[0])
        
    robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                              target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                              train_adversarially = True, Gamma = gamma_temp, budget_constraint = 'inequality')

    optimizer = torch.optim.Adam(robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
    
    # warm start base coeff
    if self.gd_FDRR_params['hidden_sizes'] == []:
        robust_mlp_model.model[0].weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
        robust_mlp_model.model[0].bias.data = torch.FloatTensor(mlp_model.intercept_.reshape(1,-1))
    else:
        robust_mlp_model.load_state_dict(mlp_model.state_dict(), strict=False)

    robust_mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = self.MLP_train_dict['epochs'], 
                                 patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                 attack_type = self.gd_FDRR_params['attack_type'])
    

    # Nominal and WC loss
    insample_loss = eval_predictions(mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
    insample_wc_loss = eval_predictions(robust_mlp_model.predict(train_temp_miss_X, self.missing_pattern[0]), trainY, self.error_metric)
        
    print(insample_loss)
    print(robust_mlp_model.best_val_loss)
    
    # Nominal, WC loss, and loss gap
    self.Loss = [insample_loss]
    self.WC_Loss = [robust_mlp_model.best_val_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.WC_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_model_ = [mlp_model]
    self.wc_node_model_ = [robust_mlp_model]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes), updated dynamically at each iteration
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    

    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_splits) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
                
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(train_temp_miss_X), trainY, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            
            if (current_node_loss > Best_loss)*(nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)

                ####### Train nominal model with missing data
                
                if self.gd_FDRR_params['hidden_sizes'] == []:
                    # Train Linear Regression model on missing data realization
                    new_mlp_model = LinearRegression(fit_intercept = True)
                    new_mlp_model.fit(train_temp_miss_X, trainY)
                else:
                    # Train Neural Network model on missing data realization
                    new_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                              target_col = self.target_features[node], fix_col = self.fixed_features[node], projection = self.gd_FDRR_params['projection'], 
                                              train_adversarially = False)
    
                    optimizer = torch.optim.Adam(new_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
    
                    new_mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                                          verbose = self.MLP_train_dict['verbose'])

                            
                retrain_loss = eval_predictions(new_mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
                wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(tensor_train_missX, temp_missing_pattern), trainY, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                                
                    solution_count = solution_count + 1
                    apply_split = True
                                        
                    # placeholders for node split
                    best_new_model = new_mlp_model
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()

            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            left_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            left_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)

            left_robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = left_target_cols, fix_col = left_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, Gamma = gamma_temp, budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                left_robust_mlp_model.model[0].weight.data = torch.FloatTensor(self.node_model_[node].coef_[0].reshape(1,-1))
                left_robust_mlp_model.model[0].bias.data = torch.FloatTensor(self.node_model_[node].intercept_.reshape(1,-1))
            else:
                left_robust_mlp_model.load_state_dict(self.node_model_[node].state_dict(), strict=False)

            optimizer = torch.optim.Adam(left_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
            
            left_robust_mlp_model.train_model(left_train_data_loader, left_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'])

            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = left_robust_mlp_model.best_val_loss

            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_model_.append(self.node_model_[node])
            self.wc_node_model_.append(left_robust_mlp_model)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)
                
            right_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)        
            right_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)

            right_robust_mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = right_target_cols, fix_col = right_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, 
                                      Gamma = gamma_temp, budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                right_robust_mlp_model.model[0].weight.data = torch.FloatTensor(best_new_model.coef_[0].reshape(1,-1))
                right_robust_mlp_model.model[0].bias.data = torch.FloatTensor(best_new_model.intercept_.reshape(1,-1))
            else:
                right_robust_mlp_model.load_state_dict(best_new_model.state_dict(), strict=False)

            # right_robust_mlp_model.load_state_dict(self.wc_node_model_[node].state_dict(), strict=False)
            # right_robust_mlp_model.W.data = self.wc_node_model_[node].W.data
            
            optimizer = torch.optim.Adam(right_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
            
            right_robust_mlp_model.train_model(right_train_data_loader, right_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'])

            # Estimate WC loss and nominal loss
            right_insample_wcloss = right_robust_mlp_model.best_val_loss
            
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())


            self.node_model_.append(best_new_model)            
            self.wc_node_model_.append(right_robust_mlp_model)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(right_fix_cols)
            self.target_features.append(right_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)

            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap_perc)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
                        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
                print(list_to_check)
            else:
                list_to_check = list_nodes_candidates

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

                            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             # print('New Node: ', node)
             # print(m0)
         
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( self.node_model_[node].predict(x0).reshape(-1))
         else:
             # WC model
             Predictions.append( self.wc_node_model_[node].predict(x0, m0).reshape(-1))

     return np.array(Predictions).reshape(-1,1)


class FiniteAdapt_Fixed(object):
  '''Finitely Adaptive Regression with Fixed Partitions at integer values/ approximates FDR from previous work
      Wrapper function that calls gd_FDRR models
      
  Paremeters:
      target_col: columns that can go missing (list)
      fix_col: columns that cannot go missing (list)
      error_metric: can be removed and passed to inner model
      **kwargs: keyword arguments to be passed to gd_FDRR model
      '''
  def __init__(self, target_col, fix_col, error_metric = 'mse', **kwargs):
      
    self.Gamma_max = len(target_col)
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.gd_FDRR_params = kwargs


  def fit(self, X, Y, val_split = 0.05, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''

    self.MLP_train_dict = kwargs
    self.FDR_models = []
    self.missing_feat_leaf = []
    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Prepare supervised learning sets
    ####### Create train/validation data loaders for torch modules
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y        
        train_temp_X = X
        valid_temp_X = X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_X = X[:-n_valid_obs]
        valid_temp_X = X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_X = torch.FloatTensor(train_temp_X)
    tensor_valid_X = torch.FloatTensor(valid_temp_X)

    train_data_loader = create_data_loader([tensor_train_X, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_X, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
    
    # Train nominal model (no missing data)
    print('Train nominal model')
    
    if self.gd_FDRR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_X, trainY)
    else:
        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = self.gd_FDRR_params['projection'], 
                                  train_adversarially = False, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])

    
    print('Start training robust models...')
    
    for gamma in np.arange(self.Gamma_max + 1):
        self.missing_feat_leaf.append(gamma)
        print(f'Budget: {gamma}')
        
        # Train robust model
        temp_fdr_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = False, 
                                  Gamma = gamma, train_adversarially = True, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(temp_fdr_model.parameters(), lr = self.MLP_train_dict['lr'])
    
        # Warm-start: use nominal model or previous iteration
        if gamma == 0:            
            if self.gd_FDRR_params['hidden_sizes'] == []:
                temp_fdr_model.model[0].weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
                temp_fdr_model.model[0].bias.data = torch.FloatTensor(mlp_model.intercept_)
            else:
                temp_fdr_model.load_state_dict(mlp_model.state_dict(), strict = False)      
            
        
        else:
            # warm-start with previous model
            temp_fdr_model.load_state_dict(self.FDR_models[-1].state_dict(), strict=False)
            
        if gamma == 0:
            self.FDR_models.append(temp_fdr_model)
            continue
        else:
            
            temp_fdr_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'], 
                                       warm_start = False, attack_type = 'random_sample')
        
            self.FDR_models.append(temp_fdr_model)
                              
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         # Total missing features in leaf
         temp_total_missing_feat = m0.sum()
         Predictions.append( self.FDR_models[temp_total_missing_feat].predict(x0).reshape(-1))
         
     return np.array(Predictions).reshape(-1,1)

class FiniteAdapt_Linear_Fixed(object):
  '''Finitely Adaptive Regression with Fixed Partitions at integer values/ approximates FDR from previous work
      Wrapper function that calls adjustable_FDR models
      
  Paremeters:
      target_col: columns that can go missing (list)
      fix_col: columns that cannot go missing (list)
      error_metric: can be removed and passed to inner model
      **kwargs: keyword arguments to be passed to adjustable_FDR model
      '''
  def __init__(self, target_col, fix_col, error_metric = 'mse', **kwargs):
      
    self.Gamma_max = len(target_col)
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.adjustable_FDR_params = kwargs


  def fit(self, X, Y, val_split = 0.05, **kwargs):
    ''' Fitting models.
        Paremeters:
            X: features
            Y: Y target
            val_split: percentage validation split (will be used for NNs)
            **kwargs: keyword arguments to be passed for training (epochs, batch size, etc.)

    '''

    self.MLP_train_dict = kwargs

    if 'weight_decay' not in list(self.MLP_train_dict.keys()):
        self.MLP_train_dict['weight_decay'] = 0

    if 'freeze_weights' not in list(self.MLP_train_dict.keys()):
        self.MLP_train_dict['freeze_weights'] = True
        
    self.FDR_models = []
    self.missing_feat_leaf = []
    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Prepare supervised learning sets
    ####### Create train/validation data loaders for torch modules
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y        
        train_temp_X = X
        valid_temp_X = X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_X = X[:-n_valid_obs]
        valid_temp_X = X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_X = torch.FloatTensor(train_temp_X)
    tensor_valid_X = torch.FloatTensor(valid_temp_X)

    train_data_loader = create_data_loader([tensor_train_X, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_X, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
    
    # Train nominal model (no missing data)
    print('Train nominal model')
    
    if self.adjustable_FDR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_X, trainY)
    else:
        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.adjustable_FDR_params['hidden_sizes'], output_size = self.adjustable_FDR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = self.adjustable_FDR_params['projection'], 
                                  train_adversarially = False, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = 1e-2, weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])

    
    print('Start training robust models...')
    
    for gamma in np.arange(self.Gamma_max + 1):
        self.missing_feat_leaf.append(gamma)
        print(f'Budget: {gamma}')
        
        # Train robust model
        temp_fdr_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.adjustable_FDR_params['hidden_sizes'], output_size = self.adjustable_FDR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = False, 
                                  Gamma = gamma, train_adversarially = True, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(temp_fdr_model.parameters(), lr = self.MLP_train_dict['lr'], 
                                     weight_decay = self.MLP_train_dict['weight_decay'])
    
        # Warm-start: use nominal model or previous iteration
        if gamma == 0:            
            if self.adjustable_FDR_params['hidden_sizes'] == []:
                temp_fdr_model.model[0].weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
                temp_fdr_model.model[0].bias.data = torch.FloatTensor(mlp_model.intercept_)
            else:
                temp_fdr_model.load_state_dict(mlp_model.state_dict(), strict = False)      
            
        
        else:
            # warm-start with previous model
            temp_fdr_model.load_state_dict(self.FDR_models[-1].state_dict(), strict=False)
            
        if gamma == 0:
            self.FDR_models.append(temp_fdr_model)
            continue
        else:
            
            temp_fdr_model.adversarial_train_model(train_data_loader, valid_data_loader, optimizer, 
                                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'],
                                       verbose = self.MLP_train_dict['verbose'], attack_type = 'random_sample', 
                                       freeze_weights = self.MLP_train_dict['freeze_weights'])
        
            self.FDR_models.append(temp_fdr_model)
                              
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         # Total missing features in leaf
         temp_total_missing_feat = m0.sum()
         Predictions.append( self.FDR_models[temp_total_missing_feat].predict(x0, m0).reshape(-1))
         
     return np.array(Predictions).reshape(-1,1)

class v2_FiniteAdapt_Linear_Fixed(object):
  '''Finitely Adaptive Regression with Fixed Partitions at integer values/ approximates FDR from previous work
      Wrapper function that calls adjustable_FDR models
      
  Paremeters:
      target_col: columns that can go missing (list)
      fix_col: columns that cannot go missing (list)
      error_metric: can be removed and passed to inner model
      **kwargs: keyword arguments to be passed to adjustable_FDR model
      '''
  def __init__(self, target_col, fix_col, error_metric = 'mse', **kwargs):
      
    self.Gamma_max = len(target_col)
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.adjustable_FDR_params = kwargs

  def fit(self, X, Y, val_split = 0.05, **kwargs):
    ''' Fitting models.
        Paremeters:
            X: features
            Y: Y target
            val_split: percentage validation split (will be used for NNs)
            **kwargs: keyword arguments to be passed for training (epochs, batch size, etc.)

    '''

    self.MLP_train_dict = kwargs
    self.FDR_models = []
    self.missing_feat_leaf = []
    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Prepare supervised learning sets
    ####### Create train/validation data loaders for torch modules
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y        
        train_temp_X = X
        valid_temp_X = X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_X = X[:-n_valid_obs]
        valid_temp_X = X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_X = torch.FloatTensor(train_temp_X)
    tensor_valid_X = torch.FloatTensor(valid_temp_X)

    train_data_loader = create_data_loader([tensor_train_X, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_X, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
    
    # Train nominal model (no missing data)
    print('Train nominal model')
    
    if self.adjustable_FDR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_X, trainY)
    else:
        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.adjustable_FDR_params['hidden_sizes'], output_size = self.adjustable_FDR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = self.adjustable_FDR_params['projection'], 
                                  train_adversarially = False, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = self.MLP_train_dict['lr'], 
                                     weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])

    
    print('Start training robust models...')
    
    for gamma in np.arange(self.Gamma_max + 1):
        self.missing_feat_leaf.append(gamma)
        print(f'Budget: {gamma}')
        
        # Train robust model
        temp_fdr_model = v2_adjustable_FDR(input_size = num_features, hidden_sizes = self.adjustable_FDR_params['hidden_sizes'], output_size = self.adjustable_FDR_params['output_size'], 
                                  target_col = self.target_col, fix_col = self.fix_col, projection = False, 
                                  Gamma = gamma, train_adversarially = True, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(temp_fdr_model.parameters(), lr = self.MLP_train_dict['lr'], 
                                     weight_decay = self.MLP_train_dict['weight_decay'])
    
        # Warm-start: use nominal model or previous iteration
        if gamma == 0:            
            if self.adjustable_FDR_params['hidden_sizes'] == []:
                temp_fdr_model.linear_correction_layer.weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
                temp_fdr_model.linear_correction_layer.bias.data = torch.FloatTensor(mlp_model.intercept_)
            else:
                temp_fdr_model.load_state_dict(mlp_model.state_dict(), strict = False)      
            
        
        else:
            # warm-start with previous model
            temp_fdr_model.load_state_dict(self.FDR_models[-1].state_dict(), strict=False)
            
        if gamma == 0:
            self.FDR_models.append(temp_fdr_model)
            continue
        else:
            
            temp_fdr_model.adversarial_train_model(train_data_loader, valid_data_loader, optimizer, 
                                       epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'],
                                       verbose = self.MLP_train_dict['verbose'], attack_type = 'random_sample', 
                                       freeze_weights = False)
        
            self.FDR_models.append(temp_fdr_model)
                              
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         # Total missing features in leaf
         temp_total_missing_feat = m0.sum()
         Predictions.append( self.FDR_models[temp_total_missing_feat].predict(x0, m0).reshape(-1))
         
     return np.array(Predictions).reshape(-1,1)
 
class Piecewise_LDR_Adaptation(object):
  '''This function initializes the GPT.
  
  Paremeters:
      D: maximum depth of the tree (should include pruning??)
      Nmin: minimum number of observations at each leaf
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      max_features: Maximum number of features to consider at each split (used for ensembles). If False, then all features are used
      **kwargs: keyword arguments to solve the optimization problem prescribed
      '''
  def __init__(self, target_col, fix_col, D = 10, Max_splits = 5, red_threshold = .01, error_metric = 'mse', **kwargs):
      
    self.D = D
    self.Max_splits = Max_splits
    self.red_threshold = red_threshold
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.gd_FDRR_params = kwargs


  def fit(self, X, Y, tree_grow_algo = 'leaf-wise', max_gap = 0.001, val_split = 0.05, **kwargs):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''

    self.MLP_train_dict = kwargs
    self.max_gap = max_gap

    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initialize root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]

    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [self.fix_col]

    # features that CAN change missing pattern within current node
    self.target_features = [self.target_col]
    
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ####### Create train/validation data loaders for torch modules
    temp_miss_X = (1-self.missing_pattern[0])*X
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))

    if val_split == 0:    
        trainY = Y
        validY = Y
        
        train_temp_miss_X = temp_miss_X
        valid_temp_miss_X = temp_miss_X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_miss_X = temp_miss_X[:-n_valid_obs]
        valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
    tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

    train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
        
    ####### Train nominal model/ no missing data/ this will remain fixed for **all** the nodes
    
    if self.gd_FDRR_params['hidden_sizes'] == []:
        # Train Linear Regression model
        mlp_model = LinearRegression(fit_intercept = True)
        mlp_model.fit(train_temp_miss_X, trainY)
    else:
        torch.manual_seed(0)

        # Train Neural Network model
        mlp_model = gd_FDRR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                  target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                                  train_adversarially = False, budget_constraint = self.gd_FDRR_params['budget_constraint'])
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
        mlp_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                              epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], 
                              verbose = self.MLP_train_dict['verbose'])
        
    ######## Train Adversarially Robust model
    # budget of robustness
    # Nominal loss: is estimated/ WC loss: set to negative value
    if self.gd_FDRR_params['budget_constraint'] == 'equality':                
        gamma_temp = 1
    elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
        gamma_temp = len(self.target_features[0])
        
    robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                              target_col = self.target_features[0], fix_col = self.fixed_features[0], projection = self.gd_FDRR_params['projection'], 
                              train_adversarially = True, 
                              Gamma = gamma_temp, budget_constraint = 'inequality')

    optimizer = torch.optim.Adam(robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'], weight_decay = self.MLP_train_dict['weight_decay'])
    
    # warm start base coeff
    if self.gd_FDRR_params['hidden_sizes'] == []:
        robust_mlp_model.model[0].weight.data = torch.FloatTensor(mlp_model.coef_[0].reshape(1,-1))
        robust_mlp_model.model[0].bias.data = torch.FloatTensor(mlp_model.intercept_.reshape(1,-1))
    else:
        robust_mlp_model.load_state_dict(mlp_model.state_dict(), strict=False)

    robust_mlp_model.sequential_train_model(train_data_loader, valid_data_loader, optimizer, 
                          epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                          attack_type = self.gd_FDRR_params['attack_type'])
    

    # Nominal and WC loss
    insample_loss = eval_predictions(mlp_model.predict(train_temp_miss_X), trainY, self.error_metric)
    insample_wc_loss = eval_predictions(robust_mlp_model.predict(train_temp_miss_X, self.missing_pattern[0]), trainY, self.error_metric)
        
    print(insample_loss)
    print(robust_mlp_model.best_val_loss)
    
    # Nominal, WC loss, and loss gap
    self.Loss = [insample_loss]
    self.WC_Loss = [robust_mlp_model.best_val_loss]
    self.Loss_gap = [self.WC_Loss[0] - self.Loss[0]]
    self.Loss_gap_perc = [(self.WC_Loss[0] - self.Loss[0])/self.Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_model_ = [mlp_model]
    self.wc_node_model_ = [robust_mlp_model]
    
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes), updated dynamically at each iteration
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    

    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_splits) or (self.Loss_gap_perc[node] <= self.max_gap):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)
                
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(train_temp_miss_X), trainY, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold)   
            
            if (current_node_loss > Best_loss)*(nominal_loss_worse_ind):    
                # Check how the robust modle performs when feature is missing
                
                wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(tensor_train_missX, temp_missing_pattern), trainY, self.error_metric)
                
                if ((wc_node_loss - current_node_loss)/wc_node_loss > self.red_threshold):
                                
                    solution_count = solution_count + 1
                    apply_split = True
                                        
                    # placeholders for node split
                    # best_new_model = new_mlp_model
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    # Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print('Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()

            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(left_target_cols)
                
            temp_miss_X = (1-left_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            left_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)        
            left_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle = False)

            left_robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = left_target_cols, fix_col = left_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, Gamma = gamma_temp, 
                                      budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                left_robust_mlp_model.model[0].weight.data = torch.FloatTensor(self.node_model_[node].coef_[0].reshape(1,-1))
                left_robust_mlp_model.model[0].bias.data = torch.FloatTensor(self.node_model_[node].intercept_.reshape(1,-1))
            else:
                left_robust_mlp_model.load_state_dict(self.node_model_[node].state_dict(), strict=False)

            optimizer = torch.optim.Adam(left_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            left_robust_mlp_model.sequential_train_model(left_train_data_loader, left_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'])

            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = left_robust_mlp_model.best_val_loss

            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(left_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_model_.append(self.node_model_[node])
            self.wc_node_model_.append(left_robust_mlp_model)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Nominal loss: is estimated/ WC loss: set to negative value
            if self.gd_FDRR_params['budget_constraint'] == 'equality':                
                gamma_temp = 1
            elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
                gamma_temp = len(right_target_cols)
            
            temp_miss_X = (1-right_missing_pattern)*X
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)
                
            right_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)        
            right_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.MLP_train_dict['batch_size'], shuffle=False)

            right_robust_mlp_model = adjustable_FDR(input_size = num_features, hidden_sizes = self.gd_FDRR_params['hidden_sizes'], output_size = self.gd_FDRR_params['output_size'], 
                                      target_col = right_target_cols, fix_col = right_fix_cols, projection = self.gd_FDRR_params['projection'], 
                                      train_adversarially = True, 
                                      Gamma = gamma_temp, budget_constraint = 'inequality')

            # warm start base coeff
            if self.gd_FDRR_params['hidden_sizes'] == []:
                right_robust_mlp_model.model[0].weight.data = torch.FloatTensor(self.node_model_[node].coef_[0].reshape(1,-1))
                right_robust_mlp_model.model[0].bias.data = torch.FloatTensor(self.node_model_[node].intercept_.reshape(1,-1))
            else:
                right_robust_mlp_model.load_state_dict(self.node_model_[node].state_dict(), strict=False)

            # right_robust_mlp_model.load_state_dict(self.wc_node_model_[node].state_dict(), strict=False)
            right_robust_mlp_model.W.data = self.wc_node_model_[node].W.data
            
            optimizer = torch.optim.Adam(right_robust_mlp_model.parameters(), lr = self.MLP_train_dict['lr'])
            
            right_robust_mlp_model.sequential_train_model(right_train_data_loader, right_valid_data_loader, optimizer, 
                                  epochs = self.MLP_train_dict['epochs'], patience = self.MLP_train_dict['patience'], verbose = self.MLP_train_dict['verbose'],
                                  attack_type = self.gd_FDRR_params['attack_type'])

            # Estimate WC loss and nominal loss
            right_insample_wcloss = right_robust_mlp_model.best_val_loss
            
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(right_insample_wcloss)
            self.Loss_gap.append( self.WC_Loss[-1] -  self.Loss[-1])
            self.Loss_gap_perc.append( (self.WC_Loss[-1] -  self.Loss[-1])/self.Loss[-1] )

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())


            self.node_model_.append(self.node_model_[node])            
            self.wc_node_model_.append(right_robust_mlp_model)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(right_fix_cols)
            self.target_features.append(right_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)

            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap_perc)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
                        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
                print(list_to_check)
            else:
                list_to_check = list_nodes_candidates

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

                            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             # print('New Node: ', node)
             # print(m0)
         
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( self.node_model_[node].predict(x0).reshape(-1))
         else:
             # WC model
             Predictions.append( self.wc_node_model_[node].predict(x0, m0).reshape(-1))

     return np.array(Predictions).reshape(-1,1)
 

 
###### Auxiliary functions

def eval_predictions(pred, target, metric = 'mae'):
    ''' Evaluates determinstic forecasts'''
    if metric == 'mae':
        return np.mean(np.abs(pred.reshape(-1)-target.reshape(-1)))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred.reshape(-1)-target.reshape(-1)).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred.reshape(-1)-target.reshape(-1))/target)
    elif metric == 'mse':
        return np.square(pred.reshape(-1)-target.reshape(-1)).mean()
    
