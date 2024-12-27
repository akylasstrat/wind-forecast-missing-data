# -*- coding: utf-8 -*-
"""
Greedy heuristic algorithm for finite adaptive regression

"""
#Import Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
#from decision_solver import *

class FiniteRetrain(object):
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
  def __init__(self, D = 10, Max_models = 5, red_threshold = .01):
      
    self.D = D
    self.Max_models = Max_models
    self.red_threshold = red_threshold
    
        
  def fit(self, X, Y, target_col, fix_col):
    ''' Function to train the Tree.
    Requires a separate function that solves the inner optimization problem, can be used with any optimization tool.
    '''       
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    # at root node, no missing data (1 = missing, 0 = not missing)    
    self.missing_pattern = [np.zeros(len(target_col))]
    # number of missing features per tree node
    self.total_missing_feat = self.missing_pattern[0].sum()

    # features that cannot change in the current node (fixed features and features that changed in previous nodes)
    self.fixed_features = [np.array(fix_col).copy().astype(int)]

    # features that *can* change missing pattern within current node
    self.target_features = [np.array(target_col).copy().astype(int)]

    self.feature = [-1]
    self.threshold = [-1]

    node_id_counter = 0
    
    # initialize root node model
    lr = LinearRegression(fit_intercept = True)
    miss_X = (1-self.missing_pattern[0])*X    
    lr.fit(miss_X, Y)
    
    ave_error = np.abs(Y.reshape(-1,1) - lr.predict(X).reshape(-1,1)).mean()
    
    self.Loss = [ave_error]
    
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.intercept_]
    self.node_model_ = [lr]
    
    self.total_models = 1
    
    for node in self.Node_id:
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_models):
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            continue
        
        # candidate features to go missing in current node        
        cand_features = self.target_features[node]
        
            
        #Initialize placeholder for subtree error
        Best_loss = self.Loss[node]
        #Check for splitting node (separate function)
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., set to 0)

            
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            miss_X = (1-temp_missing_pattern)*X    
            # fit new model
            #lr = LinearRegression(fit_intercept = True)
            #lr.fit(miss_X, Y)
            
            # check performance degradation when data are missing
            
            temp_ave_loss = np.abs( Y.reshape(-1,1) - self.node_model_[node].predict(miss_X).reshape(-1,1) ).mean()
            if (temp_ave_loss > Best_loss) * ((temp_ave_loss-self.Loss[node])/self.Loss[node] > self.red_threshold):
                
                solution_count = solution_count + 1
                apply_split = True
                                
                # fit new model
                new_lr = LinearRegression(fit_intercept = True)
                new_lr.fit(miss_X, Y)
                # placeholders for node split
                best_node_coef_ = new_lr.coef_.reshape(-1)
                best_node_bias_ = new_lr.intercept_
                best_new_model = new_lr
                best_split_feature = cand_feat
                
                Best_loss = temp_ave_loss
                # update running loss function at current node
                Best_insample_loss = np.abs( Y.reshape(-1,1) - new_lr.predict(miss_X).reshape(-1,1) ).mean()
 

        #If split is applied, update tree structure (missing feature goes to right child, left child copies current node)

        if apply_split == True:
            
            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            
            self.Loss.append(self.Loss[node])
            self.missing_pattern.append(self.missing_pattern[node].copy())
            self.node_coef_.append(self.node_coef_[node])
            self.node_bias_.append(self.node_bias_[node])
            self.node_model_.append(self.node_model_[node])
            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            self.Loss.append(Best_insample_loss)
            self.missing_pattern.append(right_missing_pattern)
            self.node_coef_.append(best_node_coef_)
            self.node_bias_.append(best_node_bias_)
            self.node_model_.append(best_new_model)
            
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)
            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     Leaf_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         Leaf_id[i] = self.Node_id[node]
     return Leaf_id
  
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
         #Go down the tree until a Leaf node is reached
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
     return np.array(Predictions)
 