# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:54:46 2024

@author: astratig
"""

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import time

def to_np(x):
    return x.detach().numpy()

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader



class MLP(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0):
        super(MLP, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        self.projection = projection
        self.UB = torch.FloatTensor([UB])
        self.LB = torch.FloatTensor([LB])
        

        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
                    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        if self.projection:
            return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB))
        else:
            return self.model(x)
            
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def predict(self, x):
        # used for inference only
        with torch.no_grad():         
            if self.projection:
                return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB)).detach().numpy()
            else:
                return self.model(x).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        for epoch in range(epochs):

            average_train_loss = self.epoch_train(train_loader, optimizer)
            val_loss = self.epoch_train(val_loader)
            '''
        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                
                # features and target for each batch
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: predict weights and combine forecasts
                output_hat = self.forward(x_batch)
                y_hat = output_hat
                
                loss_i = self.estimate_loss(y_hat, y_batch)                    
                loss = torch.mean(loss_i)
                                
                # backward pass
                loss.backward()
                optimizer.step()                
                
                running_loss += loss.item()

            average_train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            '''
            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                
                # features, wind, and generator data for each batch 
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                
                # forward pass: predict weights and combine forecasts
                output_hat = self.forward(x_batch)
                
                y_hat = output_hat
                                
                loss_i = self.estimate_loss(y_hat, y_batch)
                
                loss = torch.mean(loss_i)
                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss

def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

class resilient_MLP(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1):
        super(resilient_MLP, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        self.projection = projection
        self.UB = torch.FloatTensor([UB])
        self.LB = torch.FloatTensor([LB])
        self.target_col = torch.tensor(target_col, dtype=torch.int32)
        self.fix_col = torch.tensor(fix_col, dtype=torch.int32)

        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
    
    def pgd_linf(self, X, y, epsilon = 0.05, alpha=0.01, num_iter=30, randomize=False):
        """ Construct FGSM adversarial examples on the examples X"""
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)
            
        for t in range(num_iter):
            
            y_hat = self.forward(X + delta)
            loss = self.estimate_loss(y_hat, y).mean() 
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
            
        #print(delta.detach().numpy().max())
        
        return delta.detach()
        
        #pred = mlp_proxy(net_demand_tensor + delta)
        #loss = -nn.MSELoss()(pred, target_tensor)
        #if t % 5 == 0:
        #    print(t, loss.item())
        
        #opt.zero_grad()
        #loss.backward()
        #opt.step()
        #delta.data.clamp_(-epsilon, epsilon)

    def l_norm_attack(self, X, y, epsilon = 0.1, alpha=0.01, num_iter=15, randomize=False):

            delta = torch.zeros_like(X, requires_grad=True)
            opt = torch.optim.SGD([delta], lr=1e-2)

            for t in range(num_iter):
                
                pred = self.model(X + delta)
                loss = -nn.MSELoss()(pred, y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                delta.data.clamp_(-epsilon, epsilon)
                
                delta_proj = project_onto_l1_ball(delta, epsilon)
                
            return delta_proj.detach()


    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""

        init_alpha = torch.zeros_like(X)
        # estimate nominal loss (no missing data)
        y_hat = self.forward(X)
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and best_alpha
        wc_loss = current_loss
        best_alpha = init_alpha
        
        indexes = torch.randint(low = 0, high = X.shape[0], size = (int(perc*X.shape[0]),1))

        # loop over target columns, find worst-case loss:
        for col in self.target_col:
            alpha_temp = torch.zeros_like(X)
            alpha_temp[:,col] = 1
            
            # predict using adversarial example
            y_hat = self.forward(X*(1-alpha_temp))
            temp_loss = self.estimate_loss(y_hat, y).mean()
            
            if temp_loss > wc_loss:
                wc_loss = temp_loss
                best_alpha = alpha_temp
        
        return best_alpha
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        if self.projection:
            return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB))
        else:
            return self.model(x)
            
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    


    def adversarial_epoch_train(self, loader, opt=None):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            # find attack
            #alpha = self.missing_data_attack(X, y, gamma = 1)
            #y_hat = self.forward(X*(1-alpha))

            #delta = self.pgd_linf(X, y)
            delta = self.l_norm_attack(X,y)

            y_hat = self.forward(X + delta)
            
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def predict(self, x):
        # used for inference only
        with torch.no_grad():         
            if self.projection:
                return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB)).detach().numpy()
            else:
                return self.model(x).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        for epoch in range(epochs):

            average_train_loss = self.adversarial_epoch_train(train_loader, optimizer)
            val_loss = self.adversarial_epoch_train(val_loader)

            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return    
        
    
class gd_FDRR(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True):
        super(gd_FDRR, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        self.projection = projection
        self.UB = torch.FloatTensor([UB])
        self.LB = torch.FloatTensor([LB])
        self.target_col = torch.tensor(target_col, dtype=torch.int32)
        self.fix_col = torch.tensor(fix_col, dtype=torch.int32)
        self.gamma = Gamma
        self.train_adversarially = train_adversarially
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
        
        # # Adversarial example
        # w_coef = cp.Parameter((input_size))
        # w_bias = cp.Parameter((1))
        # x_batch = cp.Parameter((input_size))
        # y_batch = cp.Parameter((1), nonneg = True)
        
        # alpha_hat = cp.Variable((input_size), nonneg = True)
        # x_aux = cp.Variable((input_size))
        # missing_aux = cp.Variable((input_size))
        # error = cp.Variable((1))

        # Constraints = [x_aux == x_batch, error == y_batch - (w_coef@(cp.multiply(x_aux, missing_aux)) + w_bias), alpha_hat <= 1, 
        #                alpha_hat.sum() <= self.gamma, missing_aux == 1-alpha_hat] 
                        
        # objective_funct = cp.Maximize(  cp.norm(error,1) ) 
                
        # adversarial_example = cp.Problem(objective_funct, Constraints)         
        # self.adversarial_layer = CvxpyLayer(adversarial_example, parameters=[w_coef, w_bias, x_batch, y_batch], variables = [alpha_hat, error, x_aux, missing_aux])
        
        # Projection layer (can find closed-form solution)
        alpha_hat = cp.Parameter((input_size))
        alpha_proj = cp.Variable((input_size), nonneg = True)
    
        Constraints = [alpha_proj <= 1, alpha_proj.sum() <= self.gamma] 
                        
        objective_funct = cp.Minimize(  cp.norm(alpha_proj - alpha_hat) ) 
                
        l1_norm_projection = cp.Problem(objective_funct, Constraints)         
        self.projection_layer = CvxpyLayer(l1_norm_projection, parameters=[alpha_hat], variables = [alpha_proj])
    
    def l1_norm_attack(self, X, y, num_iter=10, randomize=False):
            # initialize with greedy heuristic search 
            alpha_init = self.missing_data_attack(X, y, gamma = self.gamma)
            
            alpha = torch.zeros_like(X, requires_grad=True)
            
            #alpha.data = alpha_init
            proj_simplex = nn.Softmax()
            opt = torch.optim.SGD([alpha], lr=1e-2)

            for t in range(num_iter):
                
                # pred = self.forward(X*(1-alpha))
                # loss = -self.estimate_loss(pred, y).mean() 
                
                # opt.zero_grad()
                # loss.backward()
                # opt.step()
                                    
                y_hat = self.forward(X*(1-alpha))
                loss = self.estimate_loss(y_hat, y).mean() 
                loss.backward()
                alpha.data = (alpha + 1e-2*alpha.grad.detach().sign())
                alpha.grad.zero_()
                
                # project
                # alpha_proj = project_onto_l1_ball(delta, epsilon)
                # alpha_proj = self.projection_layer(alpha)[0]
                # alpha.data = alpha_proj
                # print(alpha[0])

            return alpha.detach()
        
    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""
        
        # estimate nominal loss (no missing data)
        y_hat = self.forward(X)
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col

        # # find gamma features with higher weight        
        # for layer in self.model.children():
        #     if isinstance(layer, nn.Linear):                
        #         w_coef_param = to_np(layer.weight)
                
        # col_ind = np.argsort(np.abs(w_coef_param[0]))[::-1][:gamma]
        # for c in col_ind:   best_alpha[c] = 1

                
        for g in range(1, gamma+1):
            local_loss = []
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            alpha_init = best_alpha     
            
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                alpha_temp[:,col] = 1
                
                # predict using adversarial example
                y_hat = self.forward(X*(1-alpha_temp))
                temp_loss = self.estimate_loss(y_hat, y).mean()
                local_loss.append(temp_loss.data)
                
            best_col_ind = np.argsort(local_loss)[-1]
            best_alpha[:,current_target_col[best_col_ind]] = 1
            current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   

            #     # check if performance degrades enough, apply split
            #     if temp_loss > wc_loss:
            #         # update
            #         wc_loss = temp_loss
            #         best_alpha = alpha_temp
            #         best_col = col
            #         apply_split = True
            # #update list of eligible columns
            # if apply_split:
            #     current_target_col = torch.cat([current_target_col[0:best_col], current_target_col[best_col+1:]])   
            # else:
            #     break
        
        return best_alpha
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        return self.model(x)
        # if self.projection:
        #     return (torch.maximum(torch.minimum(self.model(x), self.UB), self.LB))
        # else:
        #     return self.model(x)
        
    
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def adversarial_epoch_train(self, loader, opt=None):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        alpha_val = np.zeros(len(self.target_col))
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):                
                w_coef_param = to_np(layer.weight)                
        col_ind = np.argsort(np.abs(w_coef_param[0]))[::-1][:self.gamma]
        for c in col_ind:   alpha_val[c] = 1


        for X,y in loader:
            
            #### Find adversarial example
            
            # Greedy top-down heuristic
            alpha = self.missing_data_attack(X, y, gamma = self.gamma)
            
            # alpha = torch.ones_like(X)*torch.FloatTensor(alpha_val).reshape(1,-1)

            # L1 attack with projected gradient descent            
            # alpha = self.l1_norm_attack(X,y)
            # # print(alpha.sum(1).mean())
            # print(alpha[0])
            # # Solve LP
            # for layer in self.model.children():
            #     if isinstance(layer, nn.Linear):
                    
            #         w_coef_param = layer.weight
            #         w_bias_param = layer.bias
            
            # alpha = self.adversarial_layer(w_coef_param, w_bias_param, X, y)
            # print(alpha[0])
            # print(alpha.sum(1).mean())
            
            # forward pass plus correction
            y_hat = self.forward(X*(1-alpha))        
            
            # y_nom = self.forward(X)
            # print(y_nom[0])
            
            #delta = self.pgd_linf(X, y)
            #y_hat = self.forward(X + delta)
                  
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)    
            loss = torch.mean(loss_i)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]

            
        return total_loss / len(loader.dataset)
    
    def predict(self, X, project = True):
        # used for inference only, returns a numpy
        with torch.no_grad():     

            if self.projection or project:
                return (torch.maximum(torch.minimum(self.model(X), self.UB), self.LB)).detach().numpy()
            else:
                return self.model(X).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, warm_start = False):
        
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        if (self.train_adversarially == False) or (warm_start):
            print('Train model for nominal case or warm-start the adversarial training')
            for epoch in range(epochs):
                
                average_train_loss = self.epoch_train(train_loader, optimizer)
                val_loss = self.epoch_train(val_loader)
    
                if (verbose != -1) and (epoch%25 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
                
        if self.train_adversarially:
            print('Start adversarial training')
            # initialize everthing
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            for epoch in range(epochs):

                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer)                
    
                if (verbose != -1)and(epoch%25 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f}")
                
                if average_train_loss < best_train_loss:
                    best_train_loss = average_train_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        return    

                # average_train_loss = self.adversarial_epoch_train(train_loader, optimizer)                
                # val_loss = self.adversarial_epoch_train(val_loader)
    
                # if (verbose != -1)and(epoch%25 == 0):
                #     print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_weights = copy.deepcopy(self.state_dict())
                #     early_stopping_counter = 0
                # else:
                #     early_stopping_counter += 1
                #     if early_stopping_counter >= patience:
                #         print("Early stopping triggered.")
                #         # recover best weights
                #         self.load_state_dict(best_weights)
                #         return    
        else:
            return

class adjustable_FDR(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1):
        super(adjustable_FDR, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        self.projection = projection
        self.UB = torch.FloatTensor([UB])
        self.LB = torch.FloatTensor([LB])
        self.target_col = torch.tensor(target_col, dtype=torch.int32)
        self.fix_col = torch.tensor(fix_col, dtype=torch.int32)
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
            
        # correction coefficients
        self.W = nn.Parameter(torch.FloatTensor(np.zeros((input_size, input_size))).requires_grad_())
                
    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""
        
        # estimate nominal loss (no missing data)
        y_hat = self.forward(X)
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col
        
        # Randomly sample number of missing features for specific batch 
        gamma_temp = torch.randint(low = 0, high = gamma, size = (1,1))
        
        for g in range(1, gamma_temp+1):
            
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            alpha_init = best_alpha     
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                alpha_temp[:,col] = 1
                
                # predict using adversarial example
                y_hat = self.forward(X*(1-alpha_temp))
                temp_loss = self.estimate_loss(y_hat, y).mean()
                
                # check if performance degrades enough, apply split
                if temp_loss > wc_loss:
                    # update
                    wc_loss = temp_loss
                    best_alpha = alpha_temp
                    best_col = col
                    apply_split = True
            # update list of eligible columns
            if apply_split:
                current_target_col = torch.cat([current_target_col[0:best_col], current_target_col[best_col+1:]])   
            else:
                break
        return best_alpha
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        
        return self.model(x)
        
    
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def adversarial_epoch_train(self, loader, opt=None):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            # find attack
            alpha = self.missing_data_attack(X, y, gamma = 1)
            
            # forward pass plus correction
          
            y_hat = self.forward(X*(1-alpha)) + torch.sum((self.W@alpha.T).T*(X*(1-alpha)), dim = 1).reshape(-1,1)
            
            y_hat_proj = (torch.maximum(torch.minimum(y_hat, self.UB), self.LB))
            
            y_nom = self.forward(X)
            #delta = self.pgd_linf(X, y)
            #y_hat = self.forward(X + delta)
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat_proj, y)                
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def predict(self, zero_imp_X, alpha):
        # used for inference only
        with torch.no_grad():     

            y_hat = self.forward(zero_imp_X) + torch.sum((self.W@alpha.T).T*(zero_imp_X), dim = 1).reshape(-1,1)

            if self.projection:
                return (torch.maximum(torch.minimum(y_hat, self.UB), self.LB)).detach().numpy()
            else:
                return y_hat
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        print('Train model normally to warm-start the adversarial training')
        for epoch in range(epochs):
            
            average_train_loss = self.epoch_train(train_loader, optimizer)
            val_loss = self.epoch_train(val_loader)

            if (verbose != -1)and(epoch%25==0):
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    break

        print('Start adversarial training')
        # re-initialize losses and epoch counter
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(epochs):

            average_train_loss = self.adversarial_epoch_train(train_loader, optimizer)
            val_loss = self.adversarial_epoch_train(val_loader)

            if (verbose != -1)and(epoch%25==0):
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return    

#% GANs

# Define the Generator          
class Generator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU(), latent_dim = 10, sigmoid_activation = True):
        super(Generator, self).__init__()

        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        
        # create sequential model
        layer_sizes = [input_size + latent_dim] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)


    def forward(self, x, noise):
        return self.model(torch.cat((x, noise), dim=1))

    def predict(self, x, noise):
        # used for inference only
        with torch.no_grad():            
            return self.model(torch.cat((x, noise), dim=1)).detach().numpy()
        
# Define the Discriminator (Critic)
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Define Wasserstein distance (critic loss)
def wasserstein_distance(real_output, fake_output):
    return - (torch.mean(real_output) - torch.mean(fake_output))

# Define the training procedure
def train_cwgan(generator, critic, optimizer_g, optimizer_c, train_features, train_targets, num_epochs, batch_size, lr):
    criterion = nn.MSELoss()
    #optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    #optimizer_c = optim.Adam(critic.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i in range(0, len(train_features), batch_size):
            # Sample batch of real data
            batch_features = train_features[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]

            # Generate fake data
            noise = torch.randn(len(batch_features), latent_dim)
            fake_targets = generator(noise)

            # Train critic
            critic_real = critic(torch.cat((batch_features, batch_targets), dim=1))
            critic_fake = critic(torch.cat((batch_features, fake_targets), dim=1))
            critic_loss = wasserstein_distance(critic_real, critic_fake)
            optimizer_c.zero_grad()
            critic_loss.backward()
            optimizer_c.step()

            # Clip critic parameters
            for p in critic.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train generator
            fake_targets = generator(noise)
            critic_fake = critic(torch.cat((batch_features, fake_targets), dim=1))
            generator_loss = -torch.mean(critic_fake)
            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch}/{num_epochs}], Generator Loss: {generator_loss.item()}, Critic Loss: {critic_loss.item()}")


