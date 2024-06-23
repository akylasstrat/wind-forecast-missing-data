# -*- coding: utf-8 -*-
"""
Plot results

@author: a.stratigakos
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

power_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0)
metadata_df = pd.read_csv('C:\\Users\\akyla\\feature-deletion-robust\\data\\smart4res_data\\wind_metadata.csv', index_col=0)

# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)
# transition matrix to generate missing data/ estimated from training data (empirical estimation)
P = np.array([[0.999, 0.001], [0.241, 0.759]])

plt.figure(constrained_layout = True)
plt.scatter(x=metadata_df['Long'], y=metadata_df['Lat'])
plt.show()

#%%

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
config['min_lag'] = 1
config['max_lag'] = 2 + config['min_lag']

target_park = 'p_1088'
pattern = 'MNAR'
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

plt.ylim([7.5, 14.25])
plt.show()


#%% LS performance degradation
 
# color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
#          'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)-LS}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)-LS}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(greedy)-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(greedy)-LS}$'}

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
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(greedy)-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(greedy)-LS}$', 
                    'FA-fixed-NN':'$\mathtt{FA(fixed)-NN}$', 
                    'FA-greedy-NN':'$\mathtt{FA(greedy)-NN}$', 
                    'FA-lin-fixed-NN':'$\mathtt{FLA(fixed)-NN}$', 
                    'FA-lin-greedy-NN':'$\mathtt{FLA(greedy)-NN}$', 
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


#%% Plotting

models = ['Pers', 'LS', 'Lasso', 'Ridge', 'LAD', 'NN', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LAD'] + [f'FinAd-LS-{n_splits}' for n_splits in Max_number_splits] + ['FinAd-NN']

models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
                    'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
                    'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$',
                    'FDRR-R':'$\mathtt{FA(const, fixed)}$',
                    'LinAdj-FDR':'$\mathtt{FA(linear, fixed)}$',
                    'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 
                    'FinAd-LS-10':'$\mathtt{FA(linear, greedy)}$', 'FinAd-LS-1':'$\mathtt{FA(linear, greedy)}$', 
                    'FinAd-NN':'$\mathtt{FinAd-NN}$'}
 
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = models
marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = mae_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models):    
    if m not in models_to_plot: continue
    else:
        y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
        x_val = temp_df['percentage'].unique().astype(float)
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
                 label = models_to_labels[m], color = colors[i], marker = marker[i])
        #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
        #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
plt.legend()
plt.ylabel('MAE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_MAE.pdf')
plt.show()


fig, ax = plt.subplots(constrained_layout = True)

temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models):    
    if m not in models_to_plot: continue
    else:
        y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
        x_val = temp_df['percentage'].unique().astype(float)
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
                 label = models_to_labels[m], color = colors[i], marker = marker[i])
        #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
        #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{target_park}_RMSE.pdf')
plt.show()

#%% Plot for a single method


 
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['LS', 'FDRR-R', 'LinAdj-FDR', 'FinAd-LS']
models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FDRR-R':'$\mathtt{FinAd(static, fixed)}$',
                    'LinAdj-FDR':'$\mathtt{FinAd(linear, fixed)}$',
                    'FinAd-LS':'$\mathtt{FinAd(linear, greedy)}$'}

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

fig, ax = plt.subplots(constrained_layout = True)

temp_df = rmse_df.query('percentage==0.01 or percentage==0.05 or percentage==0.1 or percentage==0')
std_bar = temp_df.groupby(['percentage'])[models_to_plot].std()

for i, m in enumerate(models):    
    if m not in models_to_plot: continue
    else:
        y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
        x_val = temp_df['percentage'].unique().astype(float)
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(x_val, 100*temp_df.groupby(['percentage'])[m].mean().values, 
                 label = models_to_labels[m], color = colors[i], marker = marker[i])
        #plt.fill_between(temp_df['percentage'].unique().astype(float), y_val-100*std_bar[m], 
        #                 y_val+100*std_bar[m], alpha = 0.1, color = color_list[i])    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel('Probability of failure $P_{0,1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=2, fontsize = 6)
plt.savefig(f'{cd}//plots//{target_park}_LS_RMSE.pdf')
plt.show()



#%% Testing: varying the number of missing observations/ persistence imputation
from utility_functions import *

n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
error_metric = 'rmse'
park_ids = list(power_df.columns.values)
K_parameter = np.arange(0, len(target_pred)+1)

percentage = [0, .001, .005, .01, .05, .1]
# percentage = [0, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models_to_labels = {'Pers':'$\mathtt{Imp-Pers}$', 'LS':'$\mathtt{Imp-LS}$', 
                    'Lasso':'$\mathtt{Imp-Lasso}$', 'Ridge':'$\mathtt{Imp-Ridge}$',
                    'LAD':'$\mathtt{Imp-LAD}$', 'NN':'$\mathtt{Imp-NN}$','FDRR-R':'$\mathtt{FDRR-R}$',
                    'LinAdj-FDR':'$\mathtt{LinAdj-FDR}$',
                    'FinAd-LAD':'$\mathtt{FinAd-LAD}$', 'FinAd-LS':'$\mathtt{FinAd-LS}$'}


mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
rmse_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
n_series = 4
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0

#series_missing = [c + str('_1') for c in park_ids]
#series_missing_col = [pred_col.index(series) for series in series_missing]

# Park IDs for series that could go missing
series_missing = park_ids

imputation = 'persistence'
mean_imput_values = trainPred.mean(0)
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()

config['pattern'] = 'MCAR'


for perc in percentage:
    if (config['pattern'] == 'MNAR')and(run_counter>1):
        continue

    for iter_ in range(iterations):
        
        # Dataframe to store predictions
        # temp_scale_Predictions = pd.DataFrame(data = [], columns = models)
        temp_Predictions = pd.DataFrame(data = [], columns = models)

        # Initialize dataframe to store results
        temp_df = pd.DataFrame()
        temp_df['percentage'] = [perc]
        temp_df['iteration'] = [iter_]
        
        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(testPred), len(park_ids)))
        if config['pattern'] == 'MNAR':
            P = np.array([[.999, .001], [0.2, 0.8]])
            for j, series in enumerate(series_missing):                
                # Data is MNAR, set values, control the rest within the function 
                miss_ind[:,j] = make_MNAR_chain(P, 0, len(testPred), power_df.copy()[series][split:end].values)

        else:
            P = np.array([[1-perc, perc], [0.2, 0.8]])
            for j in range(len(series_missing)):
                miss_ind[:,j] = make_chain(P, 0, len(testPred))
                #miss_ind[1:,j+1] = miss_ind[:-1,j]
                #miss_ind[1:,j+2] = miss_ind[:-1,j+1]
        
        mask_ind = miss_ind==1
        
        if run_counter%iterations==0: print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
        
        # Predictors w missing values
        miss_X = power_df[split:end].copy()[park_ids]
        miss_X[mask_ind] = np.nan
        
        miss_X = create_feat_matrix(miss_X, config['min_lag'], config['max_lag'])
        
        final_mask_ind = (miss_X.isna().values).astype(int)
        # Predictors w missing values
        miss_X_zero = miss_X.copy()
        miss_X_zero = miss_X_zero.fillna(0)
        
        # Predictors w mean imputation
        if config['impute'] != True:
            imp_X = miss_X_zero.copy()
        else:
            imp_X = miss_X.copy()
            # imputation with persistence or mean            
            if imputation == 'persistence':
                imp_X = miss_X.copy()
                # forward fill == imputation with persistence
                imp_X = imp_X.fillna(method = 'ffill')
                # fill initial missing values with previous data
                for c in imp_X.columns:
                    imp_X[c].loc[imp_X[c].isna()] = trainPred[c].mean()
                
                #for j in series_missing:
                #    imp_X[mask_ind[:,j],j] = imp_X[mask_ind[:,j],j+1]
                    
            elif imputation == 'mean':
                for j in range(imp_X.shape[1]):
                    imp_X[np.where(miss_ind[:,j] == 1), j] = mean_imput_values[j]
        
        
        #### Persistence
        pers_pred = imp_X[f'{target_park}_{min_lag}'].values.reshape(-1,1)
        temp_Predictions['Pers'] = pers_pred.reshape(-1)
        
        # if config['scale']:
        #     pers_pred = target_scaler.inverse_transform(pers_pred)            
        # pers_mae = eval_predictions(pers_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### LS model
        lr_pred = projection(lr.predict(imp_X).reshape(-1,1))
        temp_Predictions['LS'] = lr_pred.reshape(-1)
        
        # if config['scale']:
        #     lr_pred = target_scaler.inverse_transform(lr_pred)            
        # lr_mae = eval_predictions(lr_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### LASSO
        lasso_pred = projection(lasso.predict(imp_X).reshape(-1,1))
        temp_Predictions['Lasso'] = lasso_pred.reshape(-1)

        # if config['scale']:
        #     lasso_pred = target_scaler.inverse_transform(lasso_pred)    
        # lasso_mae = eval_predictions(lasso_pred, Target.values, metric= error_metric)
    
        #### RIDGE
        l2_pred = projection(ridge.predict(imp_X).reshape(-1,1))
        temp_Predictions['Ridge'] = l2_pred.reshape(-1)
        
        # if config['scale']:
        #     l2_pred = target_scaler.inverse_transform(l2_pred)    
        # l2_mae = eval_predictions(l2_pred, Target.values, metric= error_metric)
    
        #### LAD model
        lad_pred = projection(lad.predict(imp_X).reshape(-1,1))
        temp_Predictions['LAD'] = lad_pred.reshape(-1)

        # if config['scale']:
        #     lad_pred = target_scaler.inverse_transform(lad_pred)            
        # lad_mae = eval_predictions(lad_pred.reshape(-1,1), Target.values, metric=error_metric)

        #### LAD-l1 model
        # lad_l1_pred = projection(lad_l1.predict(imp_X).reshape(-1,1))
        # if config['scale']:
        #     lad_l1_pred = target_scaler.inverse_transform(lad_l1_pred)            
        # lad_l1_mae = eval_predictions(lad_l1_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        #### MLPimp
        mlp_pred = mlp_model.predict(torch.FloatTensor(imp_X.values)).reshape(-1,1)
        temp_Predictions['NN'] = mlp_pred.reshape(-1)

        # if config['scale']:
        #     mlp_pred = target_scaler.inverse_transform(mlp_pred)    
        # mlp_mae = eval_predictions(mlp_pred, Target.values, metric= error_metric)

        #### Adversarial MLP
        
        # res_mlp_pred = adj_fdr_model.predict(torch.FloatTensor(miss_X_zero.values), torch.FloatTensor(final_mask_ind)).reshape(-1,1)
        # if config['scale']:
        #     res_mlp_pred = target_scaler.inverse_transform(res_mlp_pred)    
        # res_mlp_mae = eval_predictions(res_mlp_pred, Target.values, metric= error_metric)

        #### FDRR-R (select the appropriate model for each case)
        fdr_aar_predictions = []
        for i, k in enumerate(K_parameter):
            
            fdr_pred = gd_FDRR_R_models[i].predict(torch.FloatTensor(miss_X_zero.values)).reshape(-1,1)
            # fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            fdr_pred = projection(fdr_pred)
                
            # Robust
            # if config['scale']: fdr_pred = target_scaler.inverse_transform(fdr_pred)
            fdr_aar_predictions.append(fdr_pred.reshape(-1))
        fdr_aar_predictions = np.array(fdr_aar_predictions).T
        
        # Use only the model with the appropriate K
        final_fdr_aar_pred = fdr_aar_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_aar_pred[j] = fdr_aar_predictions[j, n_miss_feat]
        final_fdr_aar_pred = final_fdr_aar_pred.reshape(-1,1)
        
        temp_Predictions['FDRR-R'] = final_fdr_aar_pred.reshape(-1)

        #### Linearly Adjustable FDR (select the appropriate model for each case)
        ladj_fdr_predictions = []
        for i, k in enumerate(K_parameter):
            
            ladj_fdr_pred = ladj_FDRR_R_models[i].predict(miss_X_zero.values, final_mask_ind).reshape(-1,1)
            # fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            ladj_fdr_pred = projection(ladj_fdr_pred)
                
            ladj_fdr_predictions.append(ladj_fdr_pred.reshape(-1))

        ladj_fdr_predictions = np.array(ladj_fdr_predictions).T
        
        # Use only the model with the appropriate K
        final_ladj_fdr_pred = ladj_fdr_predictions[:,0]
        if (perc>0)or(config['pattern']=='MNAR'):
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_ladj_fdr_pred[j] = ladj_fdr_predictions[j, n_miss_feat]
        final_ladj_fdr_pred = final_ladj_fdr_pred.reshape(-1,1)
        
        temp_Predictions['LinAdj-FDR'] = final_ladj_fdr_pred.reshape(-1)
        
        # fdr_aar_mae = eval_predictions(final_fdr_aar_pred, Target.values, metric= error_metric)

        ##### RETRAIN model
        # if config['retrain']:
        #     f_retrain_pred = retrain_pred[:,0:1].copy()
        #     if perc > 0:
        #         rows_w_missing_data = np.where(miss_X.isna().values.sum(1)==1)[0]
                
        #         for row in rows_w_missing_data:
                    
        #             temp_feat = np.sort(np.where(miss_X.isna().values[row]))[0]
        #             temp_feat = list(temp_feat)                    
        #             # find position in combinations list
        #             j_ind = retrain_comb.index(temp_feat)
        #             f_retrain_pred[row] = retrain_pred[row, j_ind]                
    
        #     retrain_mae = eval_predictions(f_retrain_pred.reshape(-1,1), Target.values, metric=error_metric)
        #     temp_df['RETRAIN'] = retrain_mae

        #### FINITE-RETRAIN-LAD and LS
        
        fin_LAD_pred = fin_LAD_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        fin_LAD_pred = projection(fin_LAD_pred)
        temp_Predictions['FinAd-LAD'] = fin_LAD_pred.reshape(-1)

        fin_LS_pred = fin_LS_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        fin_LS_pred = projection(fin_LS_pred)
        temp_Predictions['FinAd-LS'] = fin_LS_pred.reshape(-1)
        
        
        # fin_retrain_pred = fin_retrain_model.predict(miss_X_zero.values, miss_X.isna().values.astype(int))
        # fin_retrain_pred = projection(fin_retrain_pred)
        # temp_scale_Predictions['FinAd-LAD'] = fin_retrain_pred.reshape(-1)

        abs_error = np.abs(fin_LS_pred.reshape(-1,1)-Target.values)
        leaf_ind_ = fin_LS_model.apply(miss_X_zero.values, miss_X.isna().values.astype(int))

        # if config['scale']:
        #     temp_Predictions = 
        #     fin_retrain_pred = target_scaler.inverse_transform(fin_retrain_pred)            
        # fin_retrain_mae = eval_predictions(fin_retrain_pred.reshape(-1,1), Target.values, metric=error_metric)
        
        for m in models:
            temp_df[m] = [mae(temp_Predictions[m].values, Target.values)]
        mae_df = pd.concat([mae_df, temp_df])
        
        for m in models:
            temp_df[m] = [rmse(temp_Predictions[m].values, Target.values)]
        rmse_df = pd.concat([rmse_df, temp_df])
        
        run_counter += 1

pattern = config['pattern']
if config['save']:
    mae_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_MAE_results.csv')
    rmse_df.to_csv(f'{cd}\\results\\{target_park}_{pattern}_{min_lag}_steps_RMSE_results.csv')
    
