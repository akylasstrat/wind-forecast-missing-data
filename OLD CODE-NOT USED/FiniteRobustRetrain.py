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
#from decision_solver import *

class FiniteRobustRetrain(object):
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
    miss_X = (1-self.missing_pattern[0])*X    
    
    lr = QR_regressor(fit_intercept=True)
    lr.fit(miss_X, Y)
    #lr = LinearRegression(fit_intercept = True)
    #lr.fit(miss_X, Y)
    
    ave_error = np.abs(Y.reshape(-1,1) - lr.predict(X).reshape(-1,1)).mean()
    
    # nominal and WC loss
    self.Loss = [ave_error]
    self.WC_Loss = [-999]
    
    self.node_coef_ = [lr.coef_.reshape(-1)]
    self.node_bias_ = [lr.bias_]
    self.node_model_ = [lr]
    
    self.wc_node_coef_ = [lr.coef_.reshape(-1)]
    self.wc_node_bias_ = [lr.bias_]
    self.wc_node_model_ = [lr]
    
    self.total_models = 1
    
    for node in self.Node_id:
        print(f'Node = {node}')
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
            
            #new_lr = QR_regressor(fit_intercept=True)
            #new_lr.fit(miss_X, Y)

            
            # Loss at node when feature is missing with current model
            current_node_loss = np.abs( Y.reshape(-1,1) - self.node_model_[node].predict(miss_X).reshape(-1,1) ).mean()

            if self.WC_Loss[node] > 0:
                # Current node contains a robust model: train a new model and compare performance (find **best-case** improvement)
                new_lr = QR_regressor(fit_intercept=True)
                new_lr.fit(miss_X, Y)
                
                retrain_loss = np.abs( Y.reshape(-1,1) - new_lr.predict(miss_X).reshape(-1,1) ).mean()
                if node==1:
                    print(f'New:{retrain_loss}')
                    print(f'Current Loss:{current_node_loss}')
                    
                if (((current_node_loss-retrain_loss)/current_node_loss) > self.red_threshold):
                
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = retrain_loss
                    # update running loss function at current node
                    Best_insample_loss = np.abs( Y.reshape(-1,1) - new_lr.predict(miss_X).reshape(-1,1) ).mean()
  
            elif self.WC_Loss[node] < 0:
                # Current node contains a normal model, find **worst-case** loss over missing features
                
                if (current_node_loss > Best_loss) * ((current_node_loss-self.Loss[node])/self.Loss[node] > self.red_threshold):
                                        
                    solution_count = solution_count + 1
                    apply_split = True
                                    
                    # fit new model
                    new_lr = QR_regressor(fit_intercept=True)
                    new_lr.fit(miss_X, Y)

                    #new_lr = LinearRegression(fit_intercept = True)
                    #new_lr.fit(miss_X, Y)
                    # placeholders for node split
                    best_node_coef_ = new_lr.coef_.reshape(-1)
                    best_node_bias_ = new_lr.bias_
                    best_new_model = new_lr
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node
                    Best_insample_loss = np.abs( Y.reshape(-1,1) - new_lr.predict(miss_X).reshape(-1,1) ).mean() 

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print(f'Solution found, learning WC model and updating tree structure...')

            self.total_models += 2
            
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
            
            if self.inner_FDR_kwargs['budget'] == 'equality':                
                K_temp = 1
            elif self.inner_FDR_kwargs['budget'] == 'inequality':
                K_temp = len(left_target_cols)
                
            temp_miss_X = (1-self.missing_pattern[node].copy())*X
            fdr = FDR_regressor_test(K = K_temp)
            fdr.fit(temp_miss_X, Y, left_target_cols, left_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            in_sample_wc_loss = 2*fdr.objval
            
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(self.Loss[node])
            self.WC_Loss.append(in_sample_wc_loss)
            
            self.missing_pattern.append(self.missing_pattern[node].copy())
            self.node_coef_.append(fdr.coef_)
            self.node_bias_.append(fdr.bias_)
            self.node_model_.append(fdr)
            
            self.wc_node_coef_.append(self.node_coef_[node])
            self.wc_node_bias_.append(self.node_bias_[node])
            self.wc_node_model_.append(self.node_model_[node])

            
            # update tree structure
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
                
            right_fdr = FDR_regressor_test(K = K_temp)
            temp_miss_X = (1-right_missing_pattern)*X
            temp_miss_X[:,best_split_feature] = 0
            right_fdr.fit(temp_miss_X, Y, right_target_cols, right_fix_cols, **self.inner_FDR_kwargs)              
            
            # Estimate WC loss and nominal loss
            in_sample_wc_loss = 2*right_fdr.objval
            
            self.wc_node_coef_.append(right_fdr.coef_)
            self.wc_node_bias_.append(right_fdr.bias_)
            self.wc_node_model_.append(right_fdr)
            
            self.missing_pattern.append(right_missing_pattern)

            self.node_coef_.append(best_node_coef_)
            self.node_bias_.append(best_node_bias_)
            self.node_model_.append(best_new_model)
            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.Loss.append(Best_insample_loss)
            self.WC_Loss.append(-999)
            
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
         indicator = False

         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left[node] != -1) and (self.children_right[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 indicator = True
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right[node]
             #print('New Node: ', node)
         if (m0==self.missing_pattern[node]).all():
              Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
         else:
             Predictions.append( x0@self.wc_node_coef_[node] + self.wc_node_bias_[node] )
         #Predictions.append( x0@self.node_coef_[node] + self.node_bias_[node] )
   
     return np.array(Predictions)
 