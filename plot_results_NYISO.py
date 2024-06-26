# -*- coding: utf-8 -*-
"""
Plot results for NYISO data

@author: astratig
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries
from utility_functions import * 

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['save'] = False # If True, then saves models and results
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 2 # per pair of (n_nodes,percentage)
    params['pattern'] = 'MCAR'
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
config['min_lag'] = 1

nyiso_plants = ['Dutch Hill - Cohocton', 'Marsh Hill']
target_park = 'Dutch Hill - Cohocton'
config['save'] = True
min_lag = config['min_lag']
#%% Missing Not at Random
mae_df_nmar = pd.read_csv(f'{cd}\\results\\{target_park}_MNAR_{min_lag}_steps_MAE_results.csv', index_col = 0)
rmse_df_nmar = pd.read_csv(f'{cd}\\results\\{target_park}_MNAR_{min_lag}_steps_RMSE_results.csv', index_col = 0)

#%%
mae_df = pd.read_csv(f'{cd}\\results\\{target_park}_MCAR_{min_lag}_steps_MAE_results.csv', index_col = 0)
rmse_df = pd.read_csv(f'{cd}\\results\\{target_park}_MCAR_{min_lag}_steps_RMSE_results.csv', index_col = 0)

#%%
models = rmse_df.columns[:-2]

# Base performance without missing data

print((100*rmse_df.query('percentage == 0.00')[models].mean()).round(2))
print((100*mae_df.query('percentage == 0.00')[models].mean()).round(2))

#%% NMAR Plots


ls_models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']
nn_models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

fig, ax = plt.subplots(constrained_layout = True)
plt.bar(np.arange(0, 5*0.25, 0.25), 100*rmse_df_nmar[ls_models_to_plot].mean(), width = 0.2, alpha = .3, 
        yerr = 100*rmse_df_nmar[ls_models_to_plot].std())
plt.xticks(np.arange(0, 5*0.25, 0.25), ['Imp-LS', 'FA(fixed)-LS', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'], rotation = 45)

plt.bar(np.arange(1.5, 1.5+5*0.25, 0.25), 100*rmse_df_nmar[nn_models_to_plot].mean(), width = 0.2,
        yerr = 100*rmse_df_nmar[nn_models_to_plot].std())

plt.xticks(np.concatenate((np.arange(0, 5*0.25, 0.25), np.arange(1.5, 1.5+5*0.25, 0.25))), 
           ['Imp-LS', 'FA(fixed)-LS', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'] + ['Imp-NN', 'FA(fixed)-NN', 'FLA(fixed)-NN', 'FA(greedy)-NN', 'FLA(greedy)-NN'], rotation = 45)

plt.ylim([6, 13.5])
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_{min_lag}_MNAR.pdf')
plt.show()


#%% LS performance degradation
 
# color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
#          'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)-LS}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)-LS}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(learn)-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(learn)-LS}$'}

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

# colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']
line_style = ['--' '-', '-', '-']

fig, ax = plt.subplots(constrained_layout = True)



temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = 100*(temp_df.groupby(['percentage'])[models_to_plot].std())
x_val = temp_df['percentage'].unique().astype(float)

# plt.fill_between(x_val, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)

# plt.plot(temp_df['percentage'].unique().astype(float), 100*temp_df.groupby(['percentage'])['LS'].mean().values, 
#          linestyle = '--', color = 'black', label = models_to_labels['LS'])

for i, m in enumerate(models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values

    #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
    plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $p_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_{min_lag}_steps_LS_RMSE.pdf')
plt.show()



#%%
# percentage improvement
print((100* (temp_df.groupby(['percentage'])[['LS']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2).to_clipboard() )

print( (100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard())
#%% LS - sensitivity
 

models_to_plot = ['LS', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10',
                  'FA-lin-greedy-LS-25']

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 'FA-lin-greedy-LS-1':'$Q=1$', 'FA-lin-greedy-LS-5':'$Q=5$', 'FA-lin-greedy-LS-10':'$Q=10$',
                    'FA-lin-greedy-LS-25':'$Q=25$'}

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

# colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

colors = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
line_style = ['--' '-', '-', '-']

fig, ax = plt.subplots(constrained_layout = True)

x_val = temp_df['percentage'].unique().astype(float)


temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = 100*(temp_df.groupby(['percentage'])[models_to_plot].std())

# plt.fill_between(x_val, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)

# plt.plot(temp_df['percentage'].unique().astype(float), 100*temp_df.groupby(['percentage'])['LS'].mean().values, 
#          linestyle = '--', color = 'black', label = models_to_labels['LS'])

for i, m in enumerate(models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values

    #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
    plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $p_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_{min_lag}_steps_sensitivity_RMSE.pdf')
plt.show()

#%% NN performance degradation


models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)-LS}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)-LS}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(learn)-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(learn)-LS}$', 
                    'FA-fixed-NN':'$\mathtt{FA(fixed)-NN}$', 
                    'FA-greedy-NN':'$\mathtt{FA(learn)-NN}$', 
                    'FA-lin-fixed-NN':'$\mathtt{FLA(fixed)-NN}$', 
                    'FA-lin-greedy-NN':'$\mathtt{FLA(learn)-NN}$', 
                    'NN':'$\mathtt{Imp-NN}$'}

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

# colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']
line_style = ['--' '-', '-', '-']

fig, ax = plt.subplots(constrained_layout = True)

x_val = temp_df['percentage'].unique().astype(float)


temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = 100*(temp_df.groupby(['percentage'])[models_to_plot].std())

# plt.fill_between(x_val, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)

# plt.plot(temp_df['percentage'].unique().astype(float), 100*temp_df.groupby(['percentage'])['LS'].mean().values, 
#          linestyle = '--', color = 'black', label = models_to_labels['LS'])

for i, m in enumerate(models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query(f'percentage==0')['NN'].mean()
    #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
    plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $p_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6, loc = 'upper left')
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_{min_lag}_steps_NN_RMSE.pdf')
plt.show()

#%%

# percentage improvement
print((100* (temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['NN']].mean().values).round(2).to_clipboard() )

print( (100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard())
