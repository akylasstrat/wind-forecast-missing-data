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

class Linear_Correction_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        # Nominal weights
        weight = torch.Tensor(size_out, size_in)
        self.weight = torch.nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = torch.nn.Parameter(bias)

        # Linear Decision Rules
        # W = torch.zeros(size_in, size_in)
        # self.W = torch.nn.Parameter(W)  # nn.Parameter is a Tensor that's a module parameter.        
        self.W = nn.Parameter(torch.FloatTensor(np.zeros((size_in, size_in))).requires_grad_())

        
        # initialize weights and biases
        torch.nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5)) # weight init
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        # torch.nn.init.kaiming_uniform_(self.W, a=torch.math.sqrt(5)) # weight init

    def forward(self, x):
        w_times_x= torch.mm(x, self.weight.t())
        return torch.add(w_times_x, self.bias)  # w times x + b
    
    def correction_forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binaries for missing data
        """        
        # w_times_x= torch.mm(x*a, self.weights.t())
        x_imp = x*(1-a)
        # print(a.shape)
        # print(x_imp.shape)
        # print( (self.W[:,:]@a.T).shape )
        
        # print( ((self.weights@x_imp.T).T + self.bias).shape )
        # print( torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1).shape )
        # print( (((self.weights@x_imp.T).T + self.bias) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1) ).shape)

        return ((self.weight@x_imp.T).T + self.bias) + torch.sum( (self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)

        # return ((self.weight@x_imp.T).T + self.bias).reshape(-1,1) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1).tile((self.weight.shape[0],)).reshape(-1,1)


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
        # if self.projection:
        #     return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB))
        # else:
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
        if torch.is_tensor(x):
            with torch.no_grad():         
                if self.projection:
                    return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB)).detach().numpy()
                else:
                    return self.model(x).detach().numpy()
        else:
            tensor_x = torch.FloatTensor(x.copy())
            with torch.no_grad():         
                if self.projection:
                    return (torch.maximum(torch.minimum( self.model(tensor_x), self.UB), self.LB)).detach().numpy()
                else:
                    return self.model(tensor_x).detach().numpy()
            
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
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True, budget_constraint = 'inequality'):
        super(gd_FDRR, self).__init__()
        """
        Feature deletion robust regression, gradient-based training algorithm
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP 
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            budget_constraint: {'inequality', 'equality'}. If 'equality', then total number of missing features == Gamma.
                            If 'inequality', then total number of missing features <= Gamma (this is more pessimistic)
                            If 'equality', solution approximates FDRR-reformulation with inequality in the budget constraint (Gamma is upper bound).
                            This relaxes the inner max problem, which makes the robust formulation more pessimistic.
                            
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
        self.budget_constraint = budget_constraint
        
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
    
        if self.budget_constraint == 'equality':
            Constraints = [alpha_proj <= 1, alpha_proj.sum() == self.gamma] 
        elif self.budget_constraint == 'inequality':
            Constraints = [alpha_proj <= 1, alpha_proj.sum() <= self.gamma] 
                        
        objective_funct = cp.Minimize(  cp.norm(alpha_proj - alpha_hat) ) 
                
        l1_norm_projection = cp.Problem(objective_funct, Constraints)         
        self.projection_layer = CvxpyLayer(l1_norm_projection, parameters=[alpha_hat], variables = [alpha_proj])
    
    def l1_norm_attack(self, X, y, num_iter = 10, randomize=False):
            # initialize with greedy heuristic search 
            alpha_init = self.missing_data_attack(X, y, gamma = self.gamma)            
            alpha = torch.zeros_like(X[0:1], requires_grad=True)           
            
            alpha.data = alpha_init
            # proj_simplex = nn.Softmax()

            optimizer = torch.optim.SGD([alpha], lr=1e-2)
            # print(alpha)
            for t in range(num_iter):

                    
                pred = self.forward(X*(1-alpha))

                loss = -nn.MSELoss()(pred, y)
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()                
                #alpha.clamp(0,1)              
                
                # y_hat = self.forward(X*(1-alpha))
                # loss = self.estimate_loss(y_hat, y).mean() 
                # loss.backward()
                # alpha.data = (alpha + 1e-2*alpha.grad.detach().sign())
                # alpha.grad.zero_()
                
                # project
                # alpha_proj = project_onto_l1_ball(delta, epsilon)
            alpha_proj = self.projection_layer(alpha)[0]
            alpha.data = alpha_proj
            print(alpha)

            return alpha.detach()
        
    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Heuristic for fast adversarial examples of missing data on X. 
            Missing data is modeled as X*(1-alpha), if if alpha_j == 1, then x_j is missing.
            Finds binary vector alpha with greedy search.
            Args:
                X: features
                y: target
                gamma: budget of uncertainty
                self: model
            Returns: alpha which has same shape as X"""
        
        # estimate nominal loss (no missing data)
        y_hat = self.forward(X)
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss.data
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col
        
        # Iterate over gamma, greedily add one feature per iteration
        for g in range(gamma):
            
            # store losses for all features
            local_loss = []
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            # !!!!! Important to use clone/copy here, else we update both
            alpha_init = torch.clone(best_alpha)   
            
            with torch.no_grad():
                y_hat = self.forward(X*(1-best_alpha))

            # Nominal loss for this iteration using previous alpha values            
            temp_nominal_loss = self.estimate_loss(y_hat, y).mean().data
            wc_loss = temp_nominal_loss
            
            
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                # print(f'Feature:{col}')
                # print('Initial example')
                # print(alpha_temp[0])
                
                # set feature to missing
                alpha_temp[:,col] = 1
                
                # print('Temporary adversarial example')
                # print(alpha_temp[0])
                
                # predict using adversarial example
                with torch.no_grad():
                    y_adv_hat = self.forward(X*(1-alpha_temp))
                    
                temp_loss = self.estimate_loss(y_adv_hat, y).mean().data
                local_loss.append(temp_loss.data)
            
            
            
            best_col_ind = np.argsort(local_loss)[-1]                
            wc_loss = np.max(local_loss)
            best_col = current_target_col[best_col_ind]
            
            # !!!!! This approximates an equality constraint on the total budget
            if self.budget_constraint == 'equality':
                best_alpha[:,best_col] = 1
                apply_split = True
            elif self.budget_constraint == 'inequality':
                
                # check if performance degrades enough, apply split
                if wc_loss > temp_nominal_loss:
                    # update
                    best_alpha[:,best_col] = 1
                    apply_split = True
                
            
            #update list of eligible columns
            if apply_split:
                current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
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
    
    def adversarial_epoch_train_heur(self, loader, alpha, opt=None):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        index_iter_ = 0
        
        for X,y in loader:            
            #### Find adversarial example
                        
            # forward pass plus correction
            y_hat = self.forward(X*(1-alpha[index_iter_:index_iter_ + X.shape[0]]))        
            
            index_iter_ += X.shape[0]
                  
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)    
            loss = torch.mean(loss_i)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]

            
        return total_loss / len(loader.dataset)
    
    def adversarial_epoch_train(self, loader, opt=None, attack_type = 'greedy'):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        # alpha_val = np.zeros(len(self.target_col))
        # for layer in self.model.children():
        #     if isinstance(layer, nn.Linear):                
        #         w_coef_param = to_np(layer.weight)                
        # col_ind = np.argsort(np.abs(w_coef_param[0]))[::-1][:self.gamma]
        # for c in col_ind:   
        #     alpha_val[c] = 1
                    
        for X,y in loader:            
            #### Find adversarial example
            
            if attack_type == 'greedy':
                # Greedy top-down heuristic
                # Works best when budget_constraint == 'inequality'
                alpha = self.missing_data_attack(X, y, gamma = self.gamma)
            elif attack_type == 'random_sample':
                # sample random features            
                # Approximates well the FDR-reformulation with budget equality constraint
                feat_col = np.random.choice(np.arange(len(self.target_col)), replace = False, size = (self.gamma))
                alpha = torch.zeros_like(X)
                for c in feat_col: alpha[:,c] = 1
            elif attack_type == 'l1_norm':                
                # L1 attack with projected gradient descent            
                alpha = self.l1_norm_attack(X,y) 
            
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
        if torch.is_tensor(X):
            temp_X = X
        else:
            temp_X = torch.FloatTensor(X.copy())

        with torch.no_grad():     
            if (self.projection):
                return (torch.maximum(torch.minimum(self.model(temp_X), self.UB), self.LB)).detach().numpy()
            else:
                return self.model(temp_X).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, warm_start = False, 
                    attack_type = 'greedy'):
        
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
            
            # Find worst_case alpha
            
            alpha_tensor_list = []
            for X,y in train_loader:
                alpha_adv = self.missing_data_attack(X, y, gamma = self.gamma)
                alpha_tensor_list.append(alpha_adv)
                
            alpha_tensor = torch.cat(alpha_tensor_list, dim = 0)
            
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%25 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss = best_val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        self.best_val_loss = best_val_loss
                        return    
        else:
            return

class nonlinear_FDR(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True, budget_constraint = 'inequality'):
        super(nonlinear_FDR, self).__init__()
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
        self.train_adversarially = train_adversarially
        self.gamma = Gamma
        self.budget_constraint = budget_constraint
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        
        # declare nominal model
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())

        # nonlinear model to adapt to missing data inputs
        layer_sizes = [input_size] + [50] + [input_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.adapt_model = nn.Sequential(*layers)                            

    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""
        
        # estimate nominal loss (no missing data)
        # y_hat = self.forward(X)
        y_hat = self.correction_forward(X, torch.zeros_like(X, requires_grad=False))
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss.data
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col
        
        # Iterate over gamma, greedily add one feature per iteration
        for g in range(gamma):    
            # store losses for all features
            local_loss = []
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            # !!!!! Important to use clone/copy here, else we update both
            alpha_init = torch.clone(best_alpha)   
            
            y_hat = self.correction_forward(X, alpha_init)
            # y_hat = self.w@(X*(1-alpha)) + self.b + torch.sum((self.W@alpha.T).T*(X*(1-alpha)), dim = 1).reshape(-1,1)                              

            # Nominal loss for this iteration using previous alpha values            
            temp_nominal_loss = self.estimate_loss(y_hat, y).mean().data
            wc_loss = temp_nominal_loss
            
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                
                # set feature to missing
                alpha_temp[:,col] = 1
                # predict using adversarial example
                with torch.no_grad():
                    y_adv_hat = self.correction_forward(X, alpha_temp)
                    
                temp_loss = self.estimate_loss(y_adv_hat, y).mean().data
                local_loss.append(temp_loss.data)
            
            
            
            best_col_ind = np.argsort(local_loss)[-1]                
            wc_loss = np.max(local_loss)
            best_col = current_target_col[best_col_ind]
            
            # !!!!! This approximates an equality constraint on the total budget
            if self.budget_constraint == 'equality':
                best_alpha[:,best_col] = 1
                apply_split = True
            elif self.budget_constraint == 'inequality':
                # check if performance degrades enough, apply split
                if wc_loss > temp_nominal_loss:
                    # update
                    best_alpha[:,best_col] = 1
                    apply_split = True
                
            
            #update list of eligible columns
            if apply_split:
                current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
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

    def correction_forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binaries for missing data
        """
        x_imp = x*(1-a)
        # ((self.weight@x_imp.T).T + self.bias) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)

        # return ((self.w@x_imp.T).T + self.b).reshape(-1,1) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)   
        return self.model(x_imp) +  torch.sum(self.adapt_model(a)*x_imp, dim = 1).reshape(-1,1)
    
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
    

    def adversarial_epoch_train(self, loader, opt=None, attack_type = 'greedy'):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.

        for X,y in loader:            
            #### Find adversarial example

            if attack_type == 'greedy':
                # Greedy top-down heuristic
                # Works best when budget_constraint == 'inequality'
                alpha = self.missing_data_attack(X, y, gamma = self.gamma)
            elif attack_type == 'random_sample':
                # sample random features            
                # Approximates well the FDR-reformulation with budget equality constraint
                feat_col = np.random.choice(np.arange(len(self.target_col)), replace = False, size = (self.gamma))
                alpha = torch.zeros_like(X)
                for c in feat_col: alpha[:,c] = 1
            elif attack_type == 'l1_norm':                
                # L1 attack with projected gradient descent            
                alpha = self.l1_norm_attack(X,y) 
                
            # forward pass plus correction
            y_hat = self.correction_forward(X, alpha)
            loss_i = self.estimate_loss(y_hat, y)
            
            # Minimizing distance from nominal model (weights **must** be frozen, else it's degenerate)
            # y_hat = self.correction_forward(X, alpha)
            # y_nom = self.forward(X)
            # loss_i = self.estimate_loss(y_hat, y_nom)
            
            loss = torch.mean(loss_i)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]

            
        return total_loss / len(loader.dataset)
    
    def predict(self, X, alpha, project = True):
        # used for inference only
        #!!!!!! X is zero-imputed already but X*a = X, so there is no error (hopefully)
        if torch.is_tensor(X):
            temp_X = X
        else:
            temp_X = torch.FloatTensor(X.copy())

        if torch.is_tensor(alpha):
            temp_alpha = alpha
        else:
            temp_alpha = torch.FloatTensor(alpha.copy())

        with torch.no_grad():     
            y_hat = self.correction_forward(temp_X, temp_alpha)
            
            if self.projection or project:
                return (torch.maximum(torch.minimum(y_hat, self.UB), self.LB)).detach().numpy()
            else:
                return y_hat
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i

    def sequential_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = False, attack_type = 'greedy'):
        """Sequential model training:
            1. Train a nominal model.
            2. Fix model parameters, adversarial training to learn linear decision rules."""

        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        print('Train model for nominal case')
        
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
            
            print('Freeze layer weights, start adversarial training')
            if freeze_weights:
                for layer in self.model.children():
                    if isinstance(layer, nn.Linear):
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
            
            # print(self.model[0].weight)
            # print(self.model[0].bias)
                        
            # initialize everthing
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            # Find worst_case alpha
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss = best_val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        self.best_val_loss = best_val_loss
                        return    
        else:
            return

    def adversarial_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = False, attack_type = 'greedy'):
        ''' Adversarial training to learn linear decision rules.
            Assumes pre-trained weights are passed to the nominal model, only used for speed-up'''
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                    
        print('Freeze layer weights, start adversarial training')
        if freeze_weights:
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                                    
        # Find worst_case alpha
        for epoch in range(epochs):
            average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)      
            val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)

            if (verbose != -1)and(epoch%10 == 0):
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
                    self.best_val_loss = best_val_loss
                    return    


    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, warm_start = False, 
                    attack_type = 'greedy'):
        
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
            
            # Find worst_case alpha
            
            alpha_tensor_list = []
            for X,y in train_loader:
                alpha_adv = self.missing_data_attack(X, y, gamma = self.gamma)
                alpha_tensor_list.append(alpha_adv)
                
            alpha_tensor = torch.cat(alpha_tensor_list, dim = 0)
            
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
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
        else:
            return

class adjustable_FDR(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True, budget_constraint = 'inequality'):
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
        self.train_adversarially = train_adversarially
        self.gamma = Gamma
        self.budget_constraint = budget_constraint
        
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
        # self.w = nn.Parameter(torch.FloatTensor(np.zeros(input_size)).requires_grad_())
        # self.b = nn.Parameter(torch.FloatTensor(np.zeros((1))).requires_grad_())
                

    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""
        
        # estimate nominal loss (no missing data)
        # y_hat = self.forward(X)
        y_hat = self.correction_forward(X, torch.zeros_like(X, requires_grad=False))
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss.data
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col
        
        # Iterate over gamma, greedily add one feature per iteration
        for g in range(gamma):    
            # store losses for all features
            local_loss = []
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            # !!!!! Important to use clone/copy here, else we update both
            alpha_init = torch.clone(best_alpha)   
            
            y_hat = self.correction_forward(X, alpha_init)
            # y_hat = self.w@(X*(1-alpha)) + self.b + torch.sum((self.W@alpha.T).T*(X*(1-alpha)), dim = 1).reshape(-1,1)                              

            # Nominal loss for this iteration using previous alpha values            
            temp_nominal_loss = self.estimate_loss(y_hat, y).mean().data
            wc_loss = temp_nominal_loss
            
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                
                # set feature to missing
                alpha_temp[:,col] = 1
                
                # print('Temporary adversarial example')
                # print(alpha_temp[0])
                
                # predict using adversarial example
                with torch.no_grad():
                    y_adv_hat = self.correction_forward(X, alpha_temp)
                    
                temp_loss = self.estimate_loss(y_adv_hat, y).mean().data
                local_loss.append(temp_loss.data)
            
            
            
            best_col_ind = np.argsort(local_loss)[-1]                
            wc_loss = np.max(local_loss)
            best_col = current_target_col[best_col_ind]
            
            # !!!!! This approximates an equality constraint on the total budget
            if self.budget_constraint == 'equality':
                best_alpha[:,best_col] = 1
                apply_split = True
            elif self.budget_constraint == 'inequality':
                
                # check if performance degrades enough, apply split
                if wc_loss > temp_nominal_loss:
                    # update
                    best_alpha[:,best_col] = 1
                    apply_split = True
                
            
            #update list of eligible columns
            if apply_split:
                current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
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

    def correction_forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binaries for missing data
        """
        x_imp = x*(1-a)
        # ((self.weight@x_imp.T).T + self.bias) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)

        # return ((self.w@x_imp.T).T + self.b).reshape(-1,1) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)                                      
        return self.model(x_imp) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)                              
        
    
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
    

    def adversarial_epoch_train(self, loader, opt=None, attack_type = 'greedy'):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.

        for X,y in loader:            
            #### Find adversarial example

            if attack_type == 'greedy':
                # Greedy top-down heuristic
                # Works best when budget_constraint == 'inequality'
                alpha = self.missing_data_attack(X, y, gamma = self.gamma)
            elif attack_type == 'random_sample':
                # sample random features            
                # Approximates well the FDR-reformulation with budget equality constraint
                feat_col = np.random.choice(np.arange(len(self.target_col)), replace = False, size = (self.gamma))
                alpha = torch.zeros_like(X)
                for c in feat_col: alpha[:,c] = 1
            elif attack_type == 'l1_norm':                
                # L1 attack with projected gradient descent            
                alpha = self.l1_norm_attack(X,y) 
                
            # forward pass plus correction
            y_hat = self.correction_forward(X, alpha)
            loss_i = self.estimate_loss(y_hat, y)
            
            # Minimizing distance from nominal model (weights **must** be frozen, else it's degenerate)
            # y_hat = self.correction_forward(X, alpha)
            # y_nom = self.forward(X)
            # loss_i = self.estimate_loss(y_hat, y_nom)
            
            loss = torch.mean(loss_i)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]

            
        return total_loss / len(loader.dataset)
    
    def predict(self, X, alpha, project = True):
        # used for inference only
        #!!!!!! X is zero-imputed already but X*a = X, so there is no error (hopefully)
        if torch.is_tensor(X):
            temp_X = X
        else:
            temp_X = torch.FloatTensor(X.copy())

        if torch.is_tensor(alpha):
            temp_alpha = alpha
        else:
            temp_alpha = torch.FloatTensor(alpha.copy())

        with torch.no_grad():     
            y_hat = self.correction_forward(temp_X, temp_alpha)
            
            if self.projection or project:
                return (torch.maximum(torch.minimum(y_hat, self.UB), self.LB)).detach().numpy()
            else:
                return y_hat
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i

    def sequential_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = False, attack_type = 'greedy'):
        """Sequential model training:
            1. Train a nominal model.
            2. Fix model parameters, adversarial training to learn linear decision rules."""

        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        print('Train model for nominal case')
        
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
            
            print('Freeze layer weights, start adversarial training')
            if freeze_weights:
                for layer in self.model.children():
                    if isinstance(layer, nn.Linear):
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
            
            # print(self.model[0].weight)
            # print(self.model[0].bias)
                        
            # initialize everthing
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            # Find worst_case alpha
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss = best_val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        self.best_val_loss = best_val_loss
                        return    
        else:
            return

    def adversarial_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = True, attack_type = 'greedy'):
        ''' Adversarial training to learn linear decision rules.
            Assumes pre-trained weights are passed to the nominal model, only used for speed-up'''
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                    
        print('Freeze layer weights, start adversarial training')
        if freeze_weights:
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                                    
        # Find worst_case alpha
        for epoch in range(epochs):
            average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)      
            val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)

            if (verbose != -1)and(epoch%10 == 0):
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
                    self.best_val_loss = best_val_loss
                    return    


    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, warm_start = False, 
                    attack_type = 'greedy'):
        
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
            
            # Find worst_case alpha
            
            alpha_tensor_list = []
            for X,y in train_loader:
                alpha_adv = self.missing_data_attack(X, y, gamma = self.gamma)
                alpha_tensor_list.append(alpha_adv)
                
            alpha_tensor = torch.cat(alpha_tensor_list, dim = 0)
            
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
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
        else:
            return
        
class v2_adjustable_FDR(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True, budget_constraint = 'inequality'):
        super(v2_adjustable_FDR, self).__init__()
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
        self.train_adversarially = train_adversarially
        self.gamma = Gamma
        self.budget_constraint = budget_constraint
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        
        self.linear_correction_layer = Linear_Correction_Layer(layer_sizes[0], layer_sizes[1])
        
        if len(hidden_sizes) >0 :
            layers.append(activation)
            
        for i in range(len(layer_sizes) - 1):
            if i == 0: continue
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
            
        # correction coefficients
        # self.W = nn.Parameter(torch.FloatTensor(np.zeros((input_size, input_size))).requires_grad_())
        # self.w = nn.Parameter(torch.FloatTensor(np.zeros(input_size)).requires_grad_())
        # self.b = nn.Parameter(torch.FloatTensor(np.zeros((1))).requires_grad_())
                

    def missing_data_attack(self, X, y, gamma = 1, perc = 0.1):
        """ Construct adversarial missing data examples on X, returns a vector of x*(1-a)
            if a_j == 1: x_j is missing"""
        
        # estimate nominal loss (no missing data)
        # y_hat = self.forward(X)
        y_hat = self.correction_forward(X, torch.zeros_like(X, requires_grad=False))
        
        current_loss = self.estimate_loss(y_hat, y).mean()
        # initialize wc loss and adversarial example
        wc_loss = current_loss.data
        best_alpha =  torch.zeros_like(X)
        current_target_col = self.target_col
        
        # Iterate over gamma, greedily add one feature per iteration
        for g in range(gamma):    
            # store losses for all features
            local_loss = []
            # placeholders for splitting a column
            best_col = None
            apply_split = False  
            # !!!!! Important to use clone/copy here, else we update both
            alpha_init = torch.clone(best_alpha)   
            
            y_hat = self.correction_forward(X, alpha_init)
            # y_hat = self.w@(X*(1-alpha)) + self.b + torch.sum((self.W@alpha.T).T*(X*(1-alpha)), dim = 1).reshape(-1,1)                              

            # Nominal loss for this iteration using previous alpha values            
            temp_nominal_loss = self.estimate_loss(y_hat, y).mean().data
            wc_loss = temp_nominal_loss
            
            # loop over target columns (for current node), find worst-case loss:
            for col in current_target_col:
                # create adversarial example
                alpha_temp = torch.clone(alpha_init)
                
                # set feature to missing
                alpha_temp[:,col] = 1
                
                # print('Temporary adversarial example')
                # print(alpha_temp[0])
                
                # predict using adversarial example
                with torch.no_grad():
                    y_adv_hat = self.correction_forward(X, alpha_temp)
                    
                temp_loss = self.estimate_loss(y_adv_hat, y).mean().data
                local_loss.append(temp_loss.data)
            
            
            
            best_col_ind = np.argsort(local_loss)[-1]                
            wc_loss = np.max(local_loss)
            best_col = current_target_col[best_col_ind]
            
            # !!!!! This approximates an equality constraint on the total budget
            if self.budget_constraint == 'equality':
                best_alpha[:,best_col] = 1
                apply_split = True
            elif self.budget_constraint == 'inequality':
                
                # check if performance degrades enough, apply split
                if wc_loss > temp_nominal_loss:
                    # update
                    best_alpha[:,best_col] = 1
                    apply_split = True
                
            
            #update list of eligible columns
            if apply_split:
                current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
            else:
                break
        
        return best_alpha
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        x_inter = self.linear_correction_layer.forward(x) 
        return self.model(x_inter)

    def correction_forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binaries for missing data
        """
        # x_imp = x*(1-a)
        # return self.model(x_imp) + torch.sum((self.W@a.T).T*(x_imp), dim = 1).reshape(-1,1)                              
        x_inter = self.linear_correction_layer.correction_forward(x, a)  
        # print('check')
        # print(x_inter.shape)

        return self.model(x_inter)

    
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
    

    def adversarial_epoch_train(self, loader, opt=None, attack_type = 'greedy'):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss = 0.

        for X,y in loader:            
            #### Find adversarial example

            if attack_type == 'greedy':
                # Greedy top-down heuristic
                # Works best when budget_constraint == 'inequality'
                alpha = self.missing_data_attack(X, y, gamma = self.gamma)
            elif attack_type == 'random_sample':
                # sample random features            
                # Approximates well the FDR-reformulation with budget equality constraint
                feat_col = np.random.choice(np.arange(len(self.target_col)), replace = False, size = (self.gamma))
                alpha = torch.zeros_like(X)
                for c in feat_col: alpha[:,c] = 1
            elif attack_type == 'l1_norm':                
                # L1 attack with projected gradient descent            
                alpha = self.l1_norm_attack(X,y) 
                
            # forward pass plus correction
            y_hat = self.correction_forward(X, alpha)
            loss_i = self.estimate_loss(y_hat, y)
            
            # Minimizing distance from nominal model (weights **must** be frozen, else it's degenerate)
            # y_hat = self.correction_forward(X, alpha)
            # y_nom = self.forward(X)
            # loss_i = self.estimate_loss(y_hat, y_nom)

            loss = torch.mean(loss_i)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]

            
        return total_loss / len(loader.dataset)
    
    def predict(self, X, alpha, project = True):
        # used for inference only
        #!!!!!! X is zero-imputed already but X*a = X, so there is no error (hopefully)
        if torch.is_tensor(X):
            temp_X = X
        else:
            temp_X = torch.FloatTensor(X.copy())

        if torch.is_tensor(alpha):
            temp_alpha = alpha
        else:
            temp_alpha = torch.FloatTensor(alpha.copy())

        with torch.no_grad():     
            y_hat = self.correction_forward(temp_X, temp_alpha)
            
            if self.projection or project:
                return (torch.maximum(torch.minimum(y_hat, self.UB), self.LB)).detach().numpy()
            else:
                return y_hat
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i

    def sequential_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = True, attack_type = 'greedy'):
        """Sequential model training:
            1. Train a nominal model.
            2. Fix model parameters, adversarial training to learn linear decision rules."""

        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        print('Train model for nominal case')
        
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
            
            print('Freeze layer weights, start adversarial training')
            if freeze_weights:
                for layer in self.model.children():
                    if isinstance(layer, nn.Linear):
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
            
            # print(self.model[0].weight)
            # print(self.model[0].bias)
                        
            # initialize everthing
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            # Find worst_case alpha
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
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
                        self.best_val_loss = best_val_loss
                        return    
        else:
            return

    def adversarial_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, 
                               freeze_weights = True, attack_type = 'greedy'):
        ''' Adversarial training to learn linear decision rules.
            Assumes pre-trained weights are passed to the nominal model, only used for speed-up'''
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                    
        print('Freeze layer weights, start adversarial training')
        if freeze_weights:
            self.linear_correction_layer.weight.requires_grad = False
            self.linear_correction_layer.bias.requires_grad = False
            
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                                    
        # Find worst_case alpha
        for epoch in range(epochs):
            average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)      
            val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)

            if (verbose != -1)and(epoch%10 == 0):
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
                    self.best_val_loss = best_val_loss
                    return    


    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, warm_start = False, 
                    attack_type = 'greedy'):
        
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
            
            # Find worst_case alpha
            
            alpha_tensor_list = []
            for X,y in train_loader:
                alpha_adv = self.missing_data_attack(X, y, gamma = self.gamma)
                alpha_tensor_list.append(alpha_adv)
                
            alpha_tensor = torch.cat(alpha_tensor_list, dim = 0)
            
            for epoch in range(epochs):
                average_train_loss = self.adversarial_epoch_train(train_loader, optimizer, attack_type)                
                val_loss = self.adversarial_epoch_train(val_loader, None, attack_type)
    
                if (verbose != -1)and(epoch%10 == 0):
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
        else:
            return