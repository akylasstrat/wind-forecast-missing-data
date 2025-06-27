# -*- coding: utf-8 -*-
"""
Create plots and check results

@author: a.stratigakos@imperial.ac.uk
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries
from utility_functions import * 
from matplotlib.ticker import FormatStrFormatter

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

marker_dict = {
    "LR": {"marker": "x", "color": "black", 'markeredgewidth':1, 'label':'$\mathtt{Imp}$'},
    "FA-FIXED-LR": {"marker": "s", "color": "black", "markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{RF-fixed}$'},
    "FA-FIXED-LDR-LR": {"marker": "o","color": "black","markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{ARF-fixed}$'},
    "FA-LEARN-LR-10": {"marker": "s", "color": "black", "markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{RF-learn^{10}}$'},
    "FA-LEARN-LDR-LR-10": {"marker": "o","color": "black","markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{ARF-learn^{10}}$'},
    "NN": {"marker": "x", "color": "black", 'markeredgewidth':1, 'label':'$\mathtt{Imp}$'},
    "FA-FIXED-NN": {"marker": "s", "color": "black", "markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{RF-fixed}$'},
    "FA-FIXED-LDR-NN": {"marker": "o","color": "black","markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{ARF-fixed}$'},
    "FA-LEARN-NN-10": {"marker": "s", "color": "black", "markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{RF-learn^{10}}$'},
    "FA-LEARN-LDR-NN-10": {"marker": "o","color": "black","markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{ARF-learn^{10}}$'}}

models_to_labels = {'LR':'$\mathtt{LR-Imp}$', 
                    'FA-FIXED-LR':'$\mathtt{LR-RF-fixed}$',
                    'FA-FIXED-LDR-LR':'$\mathtt{LR-ARF-fixed}$',
                    'FA-LEARN-LDR-LR-10':'$\mathtt{LR-ARF-learn^{10}}$', 
                    'FA-LEARN-LDR-LR-1':'$\mathtt{LR-ARF-learn^{1}}$', 
                    'FA-LEARN-LDR-LR-5':'$\mathtt{LR-ARF-learn^{5}}$', 
                    'FA-LEARN-LDR-LR-2':'$\mathtt{LR-ARF-learn^{2}}$', 
                    'FA-LEARN-LDR-LR-20':'$\mathtt{LR-ARF-learn^{20}}$', 
                    'FA-LEARN-LR-10':'$\mathtt{LR-RF-learn^{10}}$', 
                    'FA-FIXED-NN':'$\mathtt{NN-RF-fixed}$', 
                    'FA-LEARN-NN-10':'$\mathtt{NN-RF-learn^{10}}$', 
                    'FA-FIXED-LDR-NN':'$\mathtt{NN-ARF-fixed}$', 
                    'FA-LEARN-LDR-NN-10':'$\mathtt{NN-ARF-learn^{10}}$',
                    'NN':'$\mathtt{NN-Imp}$'}


models_to_common_labels = {'LR':'$\mathtt{Imp}$', 
                    'FA-FIXED-LR':'$\mathtt{RF-fixed}$',
                    'FA-FIXED-LDR-LR':'$\mathtt{ARF-fixed}$',
                    'FA-LEARN-LDR-LR-10':'$\mathtt{ARF-learn^{10}}$', 
                    'FA-LEARN-LDR-LR-1':'$\mathtt{ARF-learn^{1}}$', 
                    'FA-LEARN-LDR-LR-5':'$\mathtt{ARF-learn^{5}}$', 
                    'FA-LEARN-LDR-LR-2':'$\mathtt{ARF-learn^{2}}$', 
                    'FA-LEARN-LDR-LR-20':'$\mathtt{ARF-learn^{20}}$', 
                    'FA-LEARN-LR-10':'$\mathtt{RF-learn^{10}}$', 
                    'FA-FIXED-NN':'$\mathtt{RF-fixed}$', 
                    'FA-LEARN-NN-10':'$\mathtt{RF-learn^{10}}$', 
                    'FA-FIXED-LDR-NN':'$\mathtt{ARF-fixed}$', 
                    'FA-LEARN-LDR-NN-10':'$\mathtt{ARF-learn^{10}}$',
                    'NN':'$\mathtt{Imp}$'}

#%% Load data results for all forecast horizons// Missing Data Completely at Random (MCAR)
config = params()

freq = '15min'
target_park = 'Noble Clinton'
config['save'] = True

all_rmse = []
steps_ = [1, 4, 8, 16]  # Forecast horizon
dataset = 'updated' # Do not change

for s in steps_:
    temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_{dataset}.csv', index_col = 0)        
    temp_df['steps'] = s    
    all_rmse.append(temp_df)

all_rmse = pd.concat(all_rmse)
print('RMSE without missing data')
print((100*all_rmse.query('P_0_1==0').groupby(['steps']).mean()).round(2))

all_rmse.to_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_all_RMSE_results_full.csv')

LR_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
NN_models_to_plot = ['NN', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN', 'FA-LEARN-NN-10', 'FA-LEARN-LDR-NN-10']

(all_rmse.groupby(['P_0_1', 'P_1_0', 'num_series', 'steps']).mean())[LR_models_to_plot + NN_models_to_plot].to_clipboard()
print((all_rmse.groupby(['P_0_1', 'P_1_0', 'num_series', 'steps']).mean())[LR_models_to_plot + NN_models_to_plot])

#%% Figure 3: RMSE vs probabilities, grid with subplots

# Select parameters for subplots
p_0_1_list = [0.05, 0.1, 0.2]
p_1_0_list = [1, 0.2, 0.1]
step_list = [1, 4, 8, 16]
base_model = 'NN'
delta_step = 0.2
markersize = 4.5
fontsize = 7

full_experiment_list = list(itertools.product(p_1_0_list, p_0_1_list))

marker_dict
(100*all_rmse.query('P_0_1>0.001 and num_series>=4').groupby(['steps', 'P_0_1', 'P_1_0', 'num_series'])[LR_models_to_plot+NN_models_to_plot].mean()).round(2).to_clipboard()

# dictionary for subplots
ax_lbl = np.arange(9).reshape(3,3)
props = dict(boxstyle='round', facecolor='white', alpha=0.3)
# axis ratio
gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

fig, ax = plt.subplot_mosaic(ax_lbl, constrained_layout = True, figsize = (7, 1.025*3), 
                             gridspec_kw = gs_kw, sharex = True, sharey = True)

# RMSE without missing data per horizon
nominal_rmse_horizon = (100*all_rmse.query('P_0_1==0').groupby(['steps'])[['LR', 'NN']].mean()).round(2)

if base_model == 'LR': models_to_plot = LR_models_to_plot
elif base_model =='NN': models_to_plot = NN_models_to_plot

for cnt, (p_1_0, p_0_1) in enumerate(full_experiment_list):
    
    temp_df = all_rmse.query(f'P_0_1=={p_0_1} and P_1_0=={p_1_0} and num_series==8 and steps in {step_list}')

    std_bar = 100*(temp_df.groupby(['steps'])[LR_models_to_plot + NN_models_to_plot].std())
    x_val = temp_df['steps'].unique().astype(int)
    y_val_horizon = 100*(temp_df.groupby(['steps'])[LR_models_to_plot + NN_models_to_plot].mean())
    
    current_axis = ax[cnt]
    plt.sca(current_axis)

    # Suptitle or Text to indicate forecasting model for each subplot
    text_title = rf'$\mathbb{{P}}_{{0,1}}={p_0_1}$'+'\n'+rf'$\mathbb{{P}}_{{1,1}}={1-p_1_0}$'  
    current_axis.text(0.65, 0.35, text_title, transform = current_axis.transAxes, 
                      fontsize=fontsize, verticalalignment='top', bbox=props)

    for k,m in enumerate(models_to_plot):
        # Line plots
        y_val = 100*temp_df.groupby(['steps'])[m].mean()
        # y_err = 100*temp_df.groupby(['steps'])[m].std()
        x_val = np.arange(len(step_list))+k*delta_step
                
        plt.plot(x_val, y_val_horizon[m].values, 
                 linestyle = '', **marker_dict[m], markersize = markersize)
        
    for l, s in enumerate(step_list):
        nom_line = current_axis.plot( np.arange(l, len(models_to_plot)*delta_step + l, delta_step), 
                                     len(models_to_plot)*[nominal_rmse_horizon.loc[s][base_model]], 
                 '--', color = 'black', markersize = markersize, 
                 label = rf'$\mathtt{{{base_model}}}$')
        nom_line[0].set_dashes([3,1])

    plt.xticks(np.arange(len(step_list))+0.25, step_list)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# fig.set_xticks(x_val, np.arange(5))
ysuplabel = fig.supylabel('RMSE (%)')
xsuplabel = fig.supxlabel(r'Forecast Horizon $h$ (15 minutes)')
plt.ylim([1.5, 20.5])

label_list = [rf'$\mathtt{{{base_model}}}-$' + l for l in labels[:len(models_to_plot)]] + [rf'$\mathtt{{{base_model}}}$ (no missing data)']

lgd = fig.legend(lines[:len(models_to_plot)+1], label_list, fontsize=fontsize, ncol = 1, loc = (1, .8), 
                 bbox_to_anchor=(1, 0.58), labelspacing = 1.1)

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{base_model}_RMSE_MCAR_mat_{dataset}_wide.pdf',  
                bbox_extra_artists=(lgd,ysuplabel,xsuplabel), bbox_inches='tight')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{base_model}_RMSE_MCAR_mat_{dataset}_wide.png',  
                bbox_extra_artists=(lgd,ysuplabel,xsuplabel), bbox_inches='tight')
plt.show()

#%% Same plot, subset of methods, Figure used for presentations

temp_marker_dict = {
    "LR": {"marker": "x", "color": "black", 'markeredgewidth':1.25, 'label':'$\mathtt{Imp}$'},
    "FA-LEARN-LDR-LR-10": {"marker": "o","color": "tab:blue","markerfacecolor": "none",'markeredgewidth':1.25, 'label':'$\mathtt{ARF(learn^{10})}$'},}

# Select parameters for subplots
p_0_1_list = [0.05, 0.1, 0.2]
p_1_0_list = [1, 0.2, 0.1]
step_list = [1, 4, 8, 16]
base_model = 'LR'
delta_step = 0.4
markersize = 5
fontsize = 7

# LR_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
LR_models_to_plot = ['LR', 'FA-LEARN-LDR-LR-10']

full_experiment_list = list(itertools.product(p_1_0_list, p_0_1_list))

marker_dict
(100*all_rmse.query('P_0_1>0.001 and num_series>=4').groupby(['steps', 'P_0_1', 'P_1_0', 'num_series'])[LR_models_to_plot+NN_models_to_plot].mean()).round(2).to_clipboard()

# dictionary for subplots
ax_lbl = np.arange(9).reshape(3,3)
props = dict(boxstyle='round', facecolor='white', alpha=0.3)
# axis ratio
gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

fig, ax = plt.subplot_mosaic(ax_lbl, constrained_layout = True, figsize = (5.86, 3.46), 
                             gridspec_kw = gs_kw, sharex = True, sharey = True)

# RMSE without missing data per horizon
nominal_rmse_horizon = (100*all_rmse.query('P_0_1==0').groupby(['steps'])[['LR', 'NN']].mean()).round(2)

if base_model == 'LR': models_to_plot = LR_models_to_plot
elif base_model =='NN': models_to_plot = NN_models_to_plot

for cnt, (p_1_0, p_0_1) in enumerate(full_experiment_list):
    
    temp_df = all_rmse.query(f'P_0_1=={p_0_1} and P_1_0=={p_1_0} and num_series==8 and steps in {step_list}')

    std_bar = 100*(temp_df.groupby(['steps'])[LR_models_to_plot + NN_models_to_plot].std())
    x_val = temp_df['steps'].unique().astype(int)
    y_val_horizon = 100*(temp_df.groupby(['steps'])[LR_models_to_plot + NN_models_to_plot].mean())
    
    current_axis = ax[cnt]
    plt.sca(current_axis)

    # Suptitle or Text to indicate forecasting model for each subplot
    text_title = rf'$\mathbb{{P}}_{{0,1}}={p_0_1}$'+'\n'+rf'$\mathbb{{P}}_{{1,1}}={1-p_1_0}$'  
    current_axis.text(0.65, 0.35, text_title, transform = current_axis.transAxes, 
                      fontsize=fontsize, verticalalignment='top', bbox=props)

    for k,m in enumerate(models_to_plot):
        # Line plots
        y_val = 100*temp_df.groupby(['steps'])[m].mean()
        x_val = np.arange(len(step_list))+k*delta_step
        plt.plot(x_val, y_val_horizon[m].values, 
                 linestyle = '', **temp_marker_dict[m], markersize = markersize)
        
    for l, s in enumerate(step_list):
            
        nom_line = current_axis.plot( np.arange(l, len(models_to_plot)*delta_step + l, delta_step), 
                                     len(models_to_plot)*[nominal_rmse_horizon.loc[s][base_model]], 
                 '--', color = 'black', markersize = markersize, 
                 label = rf'$\mathtt{{{base_model}}}$')
        nom_line[0].set_dashes([3,1])

    plt.xticks(np.arange(len(step_list))+0.25, step_list)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# fig.set_xticks(x_val, np.arange(5))
ysuplabel = fig.supylabel('RMSE (%)')
xsuplabel = fig.supxlabel(r'Forecast Horizon $h$ (15 minutes)')
plt.ylim([1.5, 20.5])

label_list = [rf'$\mathtt{{{base_model}}}-$' + l for l in labels[:len(models_to_plot)]] + [rf'$\mathtt{{{base_model}}}$ (no missing data)']

lgd = fig.legend(lines[:len(models_to_plot)+1], label_list, fontsize=fontsize, ncol = 3, loc = (1, .8), 
                 bbox_to_anchor=(0.25, -0.075), labelspacing = 1.1)

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{base_model}_RMSE_MCAR_mat_{dataset}_ESIG_wide.pdf',  
                bbox_extra_artists=(lgd,ysuplabel,xsuplabel), bbox_inches='tight')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{base_model}_RMSE_MCAR_mat_{dataset}_ESIG_wide.png',  
                bbox_extra_artists=(lgd,ysuplabel,xsuplabel), bbox_inches='tight')
plt.show()

#%% Sensitivity to number of subsets Q// connected scatterplot (does not appear in the paper)

all_rmse = []
steps_ = [1, 4, 8, 16]
min_lag = 1

for s in steps_:
    temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_{dataset}.csv', index_col = 0)
    temp_df['steps'] = s    
    all_rmse.append(temp_df)

all_rmse = pd.concat(all_rmse)
print('RMSE without missing data')
print((100*all_rmse.query('P_0_1==0').groupby(['steps']).mean()).round(2))

temp_df = all_rmse.query(f'P_0_1=={0.2} and P_1_0=={0.1} and num_series==8 and steps=={min_lag}')

# Load trained models
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_NN_models_dict = pickle.load(handle)               
    
model_dictionary = FA_LEARN_LDR_LR_models_dict

Q_list = np.array(list(model_dictionary.keys()))
Q_list = np.sort(Q_list)

### Find WC performance gap
WC_gap = []
WC_gap_dict = {}
abs_gap = []

max_UB = []
best_LB = []

# Upper and lower bound (RMSE) at index with worst-case gap
UB_wc_ind = []
LB_wc_ind = []

for q in Q_list:
    temp_model = model_dictionary[q]
    leaf_ind = np.where(np.array(temp_model.feature) == -1)[0]
        
    WC_gap.append(np.array(temp_model.Loss_gap_perc)[leaf_ind].max())
    WC_gap_dict[q] = np.array(temp_model.Loss_gap_perc)[leaf_ind].max()

    abs_gap.append(np.array(temp_model.Loss_gap)[leaf_ind].max())
    max_UB.append(np.array(temp_model.UB_Loss)[leaf_ind].max())
    best_LB.append(np.array(temp_model.LB_Loss)[leaf_ind].max())
    # find leaf index with highest gap
    ind_wc = np.argmax(np.array(temp_model.Loss_gap_perc)[leaf_ind])
    UB_wc_ind.append(100*np.sqrt(np.array(temp_model.UB_Loss)[leaf_ind][ind_wc]))
    LB_wc_ind.append(100*np.sqrt(np.array(temp_model.LB_Loss)[leaf_ind][ind_wc]))

UB_wc_ind = np.array(UB_wc_ind)
LB_wc_ind = np.array(LB_wc_ind)
WC_gap = np.array(WC_gap)

### Sensitivity analysis table results
models_to_plot = ['LR', 'FA-FIXED-LDR-LR','FA-LEARN-LDR-LR-1', 'FA-LEARN-LDR-LR-2', 'FA-LEARN-LDR-LR-5', 
                  'FA-LEARN-LDR-LR-10', 'FA-LEARN-LDR-LR-20']

print( 100*all_rmse.query(f'P_0_1 == 0.2 and P_1_0 == 0.1 and steps == {min_lag}').mean()[models_to_plot] )
print('WC Gap, percentage')
print((np.array(WC_gap)).round(2))

text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'tab:red', 
                  alpha=0.5)


### Absolute gap plot
fig, axes = plt.subplots(constrained_layout = True, nrows = 1, sharex = True, 
                         sharey = False, figsize = (3.5, 1.5))
plt.plot(Q_list, np.array(WC_gap), linestyle = '-', marker = '+')
plt.xticks(Q_list, Q_list)
plt.xlabel('Q')
plt.ylabel('RMSE (%)')
plt.show()

### Sensitivity plot/ connected scatterplot
Q_values_to_plot = [ 1,  2,  5, 10, 20, 50, 100]
base_model = 'LR'
model_list = [f'FA-LEARN-LDR-{base_model}-{q}' for q in Q_values_to_plot]

fig, ax1 = plt.subplots(constrained_layout = True, figsize = (3.5, 1.5))

yval = 100*temp_df.mean()[model_list].values
xval = -WC_gap


text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'black', alpha=0.5)

plt.plot(xval, yval, marker = 'o', color = 'black', markerfacecolor = 'white', markeredgewidth = 1, label = '$\mathtt{ARF-learn^{10}}$', 
         linestyle='-')

# Text to indicate forecasting model for each subplot
for i, q in enumerate(Q_values_to_plot):
    if q == 20: continue
    plt.text(xval[i]-2.5, yval[i] + 2.5, f'$Q={q}$', fontsize=6, verticalalignment='top', bbox=text_props, color = 'black')
    
plt.ylabel('RMSE (%)', fontsize = 7)
plt.xlabel('Max. $\mathtt{RelGap}$ (%)', fontsize = 7)
plt.xticks(-np.arange(0, 70, 10), np.arange(0, 70, 10))
# plt.ylim([2.5, 20])
plt.ylim([5, 20.5])

fig.tight_layout()  # otherwise the right y-label is slightly clipped

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
ax1.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='tab:green', marker = '8', label = '$\mathtt{LR-ARF-learn}^{Q}$', linewidth = 1)

ax1.tick_params(axis='y')
ax1.set_xticks(range(5), [1, 2, 5, 10, 20])

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.1, -0.05))

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel(r'Norm. Max. Gap (%)', color=color)  # we already handled the x-label with ax1
ax2.plot((np.array(WC_gap)).round(2), '-.', color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%%%%%% Figures showing weights and partitions

all_rmse = []
steps_ = [1]
min_lag = 1

with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_FIXED_LDR_LR_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LDR_LR_model = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)           

plant_ids = ['Marble River', 'Noble Clinton', 'Noble Ellenburg',
             'Noble Altona', 'Noble Chateaugay', 'Jericho Rise', 'Bull Run II Wind', 'Bull Run Wind']

target_model = FA_LEARN_LDR_LR_models_dict[10]
fixed_model = FA_FIXED_LDR_LR_model

target_node = 0
leaf_ind = np.where(np.array(target_model.feature)==-1)

print(f'Is the current node a leaf: {target_node in leaf_ind[0]}')
print(f'Feature selected for split: {target_model.feature[target_node]}')

largest_magn_ind = np.argmax(np.abs(target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1)))
largest_pos_ind = np.argmax((target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1)))

print(f'Feature with highest absolute weight: {largest_magn_ind}')
print(f'Feature with highest positive weight: {largest_pos_ind}')

n_feat = len(target_model.target_features[0]) + len(target_model.fixed_features[0])

### w_opt vs D (not included in the paper)
w_opt = target_model.node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
w_adv = target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
D = target_model.wc_node_model_[target_node].model[0].W.detach().numpy()
D_wc_row = target_model.wc_node_model_[target_node].model[0].W.detach().numpy()[:,target_model.feature[target_node]]
#!!!!! Note, D_[i,:] represents the linear correction terms when the i-th feature is missing// the formulation has the transpose

plant_list = [f'Plant {i+1}' for i in range(8)]  # X-axis (Farms)
time_lags = ['t', 't-1', 't-2']  # Y-axis (Lags)

##### Figure 6: Linear correction terms, Heatmap
feat_names = plant_list + ['Weather']

### Plot w_opt for two subsets (Fig. 4)
fig, ax = plt.subplots(constrained_layout = True, ncols = 1, nrows = 1, figsize = (3.5, 2.25))

# Create Heatmap

vmin = D.min()
vmax = D.max()
v_abs_max = np.maximum(np.abs(vmin), np.abs(vmax))

cax = ax.imshow(D[::-1], aspect='auto', cmap='PiYG', interpolation='nearest', 
                vmin = -v_abs_max, vmax = v_abs_max)

# Colorbar
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Magnitude", fontsize=7)

plant_list = [f'Plant {i+1}' for i in range(8)]  # X-axis (Farms)
plant_list.insert(1, r"$\bigstar$")

# Heatmap of linear correction terms
feat_names = plant_list + ['Weather']

y_tick_ind = [23, 21,20, 17, 14, 11, 8, 5, 2, 0]
x_tick_ind = [1, 3, 4, 7, 10, 13, 16, 19, 22, 24]

plt.yticks(y_tick_ind, feat_names)
plt.xticks(x_tick_ind, feat_names, rotation = 45)

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_heatmap_linear_corrections.pdf', 
            bbox_inches='tight')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_heatmap_linear_corrections.png', 
            bbox_inches='tight')

plt.show()

#%% Figure 4: Optimistic weights for first two splits, checking partitions across horizons

all_rmse = []
steps_ = [1]
min_lag = 1
target_node = 0

with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_FIXED_LDR_LR_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LDR_LR_model = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)           

plant_ids = ['Marble River', 'Noble Clinton', 'Noble Ellenburg',
             'Noble Altona', 'Noble Chateaugay', 'Jericho Rise', 'Bull Run II Wind', 'Bull Run Wind']

target_ldr_lr_model = FA_LEARN_LDR_LR_models_dict[10]
target_lr_model = FA_LEARN_LR_models_dict[10]

fixed_model = FA_FIXED_LDR_LR_model

####### Figure 4: Optimistic weights for first two splits

target_model = FA_LEARN_LDR_LR_models_dict[10]

plant_list = [f'Plant {i+1}' for i in range(8)]  # X-axis (Farms)
time_lags = ['t', 't-1', 't-2']  # Y-axis (Lags)

### Plot w_opt for two subsets (Fig. 4)
fig, axes = plt.subplots(constrained_layout = True, ncols = 2, nrows = 1, sharex = False, 
                         sharey = True, figsize = (3.5, 2.25))

height_ = 0.61
delta_step = 0.3

text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'white', 
                  alpha=0.25)
arrow_props = dict(arrowstyle="->", linewidth=0.7)

for col, target_node in enumerate([0,2]):
    index = target_model.feature[target_node]

    leaf_ind = np.where(np.array(target_model.feature)==-1)
    split_feat = target_model.feature[target_node]
    print(f'Is the current node a leaf: {target_node in leaf_ind[0]}')
    print(f'Feature selected for split: {target_model.feature[target_node]}')
    
    largest_magn_ind = np.argmax(np.abs(target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1)))
    largest_pos_ind = np.argmax((target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1)))
    
    print(f'Feature with highest absolute weight: {largest_magn_ind}')
    print(f'Feature with highest positive weight: {largest_pos_ind}')
    
    n_feat = len(target_model.target_features[0]) + len(target_model.fixed_features[0])
    ### Coefficients

    w_opt = target_model.node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
    bias_opt = target_model.node_model_[target_node].model[0].bias.detach().numpy().reshape(-1)
    w_adv = target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
    bias_adv = target_model.wc_node_model_[target_node].model[0].bias.detach().numpy().reshape(-1)
    D = target_model.wc_node_model_[target_node].model[0].W.detach().numpy()
    D_wc_row = target_model.wc_node_model_[target_node].model[0].W.detach().numpy()[:,index]

    current_ax = axes[col]
    plt.sca(current_ax)

    for i in range(0, 24, 3):
        t_i = np.arange(i, i+3)
        plt.barh( t_i[0] + delta_step, w_opt[t_i[0]], height = height_, color = 'black')
        plt.barh( t_i[1], w_opt[t_i[1]], height =height_, color = 'black')
        plt.barh( t_i[2] - delta_step, w_opt[t_i[2]], height = height_, color = 'black')
    
    plt.barh(25, w_opt[-1], height = height_, color = 'black')
    plt.barh(28, bias_opt, height = height_, color = 'black')
    
    # plt.title(fr'$\mathbf{{w}}^{{\text{{opt}}}}_{{{target_node}}}$')
    plt.title(fr'$\mathcal{{U}}_{target_node}: \mathbf{{w}}^{{\text{{opt}}}}$')
    plt.xlabel('Magnitude')

    # Marker to show selected feature
    plt.scatter(1.1*w_opt[target_model.feature[target_node]], target_model.feature[target_node] + delta_step, color = 'black', 
                marker = '*')
    
    if target_node == 0:
        axes[0].annotate('$\mathtt{t}$', xy=(0.0, 21.25), xytext=(0.75, 20),
                    arrowprops=arrow_props, bbox=text_props, fontsize = 5)
        
        axes[0].annotate('$\mathtt{t-1}$', xy=(0.1, 22), xytext=(0.75, 21.75),
                    arrowprops=arrow_props, bbox=text_props, fontsize = 5)
        
        axes[0].annotate('$\mathtt{t-2}$', xy=(0.0, 22.75), xytext=(0.75, 24),
                    arrowprops=arrow_props, bbox=text_props, fontsize = 5)
    
    plt.xlim([-1.1,1.5])
    
plt.yticks(list(range(1,25,3))+[25, 28], plant_list + ['Weather', 'Bias'])

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_weight_opt_barplot_nodes.pdf')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_weight_opt_barplot_nodes.png')
plt.show()

#%%
######### Fig. 7
### For a given subset, plot w_adv for ARF RF model
### Plot w_opt for two subsets

plant_ids = ['Marble River', 'Noble Clinton', 'Noble Ellenburg',
             'Noble Altona', 'Noble Chateaugay', 'Jericho Rise', 'Bull Run II Wind', 'Bull Run Wind']
feat_names = plant_list + ['Weather', 'Bias']

min_lag = [1]
horizon_model_dict = {}
weight_mat_dict = {}

n_rows, n_cols = 4, 1
fig_width = 3.5
fig_height = 2.1*(n_rows / n_cols)  # Preserve aspect ratio
# fig_height = 3.5

# Titles
h_values = [1, 4, 8, 16]

weights_flattened = []
# Find weight matrices per horizon
for m in min_lag:
    with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{m}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           

target_ARF_model = FA_LEARN_LDR_LR_models_dict[10]
target_RF_model = FA_LEARN_LR_models_dict[10]

target_node = 0

fig, axes = plt.subplots(constrained_layout = True, ncols = 2, nrows = 1, sharex = False, 
                         sharey = True, figsize = (3.5, 2.25))

height_ = 0.61
delta_step = 0.3

text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'white', 
                  alpha=0.25)
arrow_props = dict(arrowstyle="->", linewidth=0.7)

# Tree parameters
index = target_ARF_model.feature[target_node]

leaf_ind = np.where(np.array(target_ARF_model.feature)==-1)
split_feat = target_ARF_model.feature[target_node]
print(f'Is the current node a leaf: {target_node in leaf_ind[0]}')
print(f'Feature selected for split: {target_model.feature[target_node]}')

n_feat = len(target_ARF_model.target_features[0]) + len(target_ARF_model.fixed_features[0])

## RF model
RF_w_adv = target_RF_model.wc_node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
RF_bias_adv = target_RF_model.wc_node_model_[target_node].model[0].bias.detach().numpy().reshape(-1)

current_ax = axes[0]
plt.sca(current_ax)
for i in range(0, 24, 3):
    t_i = np.arange(i, i+3)
    plt.barh( t_i[0] + delta_step, RF_w_adv[t_i[0]], height = height_, color = 'black')
    plt.barh( t_i[1], RF_w_adv[t_i[1]], height =height_, color = 'black')
    plt.barh( t_i[2] - delta_step, RF_w_adv[t_i[2]], height = height_, color = 'black')
plt.barh(25, RF_w_adv[-1], height = height_, color = 'black')
plt.barh(28, RF_bias_adv, height = height_, color = 'black')

plt.title(fr'$\mathtt{{RF}}: \mathbf{{w}}^{{\text{{adv}}}}$')
plt.xlabel('Magnitude')
plt.xlim([-1.1,1.5])

## RF model
ARF_w_adv = target_ARF_model.wc_node_model_[target_node].model[0].weight.detach().numpy().reshape(-1)
ARF_bias_adv = target_ARF_model.wc_node_model_[target_node].model[0].bias.detach().numpy().reshape(-1)
D = target_ARF_model.wc_node_model_[target_node].model[0].W.detach().numpy()
D_wc_row = target_ARF_model.wc_node_model_[target_node].model[0].W.detach().numpy()[:,index]

# Fix alpha adversarial with 0 everywhere and 1 at split feature
alpha_adv = np.zeros(ARF_w_adv.shape)
alpha_adv[split_feat] = 1
w_adv_corrected = (1-alpha_adv)*ARF_w_adv + alpha_adv*D_wc_row

current_ax = axes[1]
plt.sca(current_ax)

for i in range(0, 24, 3):
    t_i = np.arange(i, i+3)
    plt.barh( t_i[0] + delta_step, w_adv_corrected[t_i[0]], height = height_, color = 'black')
    plt.barh( t_i[1], w_adv_corrected[t_i[1]], height =height_, color = 'black')
    plt.barh( t_i[2] - delta_step, w_adv_corrected[t_i[2]], height = height_, color = 'black')
    
plt.barh(25, w_adv_corrected[-1], height = height_, color = 'black')
plt.barh(28, ARF_bias_adv, height = height_, color = 'black')

# plt.title(fr'$\mathtt{{ARF}}: \mathbf{{w}}^{{\text{{adv}}}} + \mathbf{{\alpha}}^{{\text{{adv}}^{{\top}} }} \mathbf{{D}}^{{\text{{adv}}}}_{{[{index},:]}}$')
plt.title(fr'$\mathtt{{ARF}}: \mathbf{{w}}^{{\text{{adv}}}} + \mathbf{{D}}^{{\text{{adv}}}}_{{[:,{index}]}}$')
plt.xlabel('Magnitude')
plt.xlim([-1.1,1.5])
        
plt.yticks(list(range(1,25,3))+[25, 28], plant_list + ['Weather', 'Bias'])

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_weight_adv_barplot.pdf')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_weight_adv_barplot.png')
plt.show()

#%%%%% Heatmaps with optimistic weights across final partition 

all_rmse = []
steps_ = [1]
min_lag = 1
target_node = 1

with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_FIXED_LDR_LR_model_weather.pickle', 'rb') as handle:
    FA_FIXED_LDR_LR_model = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)           

plant_ids = ['Marble River', 'Noble Clinton', 'Noble Ellenburg',
             'Noble Altona', 'Noble Chateaugay', 'Jericho Rise', 'Bull Run II Wind', 'Bull Run Wind']

target_model = FA_LEARN_LR_models_dict[10]

leaf_ind = np.where(np.array(target_model.feature)==-1)
parent_node = target_model.parent_node[target_node]
n_feat = len(target_model.target_features[0]) + len(target_model.fixed_features[0])

weight_matrix = np.zeros((len(leaf_ind[0]), 26))

for i, leaf in enumerate(leaf_ind[0]):
    
    ### Coefficients
    w_opt = target_model.node_model_[leaf].model[0].weight.detach().numpy().reshape(-1)
    bias_opt = target_model.node_model_[leaf].model[0].bias.detach().numpy().reshape(-1)

    weight_matrix[i,:-1] = w_opt
    weight_matrix[i,-1] = bias_opt

feat_names = plant_list + ['Weather', 'Bias']

### Plot w_opt for two subsets (Fig. 4)
# Display heatmaps
vmin = weight_matrix.min()
vmax = weight_matrix.max()
v_abs_max = np.maximum(np.abs(vmin), np.abs(vmax))

fig, ax = plt.subplots(constrained_layout = True, ncols = 1, nrows = 1, figsize = (3.5, 2.25))

# Create Heatmap
cax = ax.imshow(weight_matrix.T[::-1], aspect='auto', cmap='PiYG', 
                interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)

# Colorbar
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Magnitude", fontsize=7)

# Annotate split feature
for i, leaf in enumerate(leaf_ind[0]):
    # missing pattern    
    missing_pattern = target_model.missing_pattern[leaf]
    miss_ind = np.where(missing_pattern==1)[0]

    # plot fixed patterns 
    fixed_ind = target_model.fixed_features[leaf]
    fixed_ind = np.array([s for s in fixed_ind if s not in list(miss_ind)] + [25])
    
    plt.plot( len(fixed_ind)*[i], 25-fixed_ind, '+', color = 'black', markersize = 5)
        
    if (missing_pattern==1).sum() == 0: 
        continue
    else: 
        plt.plot( len(miss_ind)*[i], 25-miss_ind, '_', color = 'black', markersize = 5)



y_tick_ind = list(range(1,25,3))+[25, 26]
y_tick_ind = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
y_tick_ind = [24, 21, 18, 15, 12, 9, 6, 3, 1, 0]

plt.yticks(y_tick_ind, feat_names)

xtick_labels = [r'$\mathcal{U}_{%d}$' % node_id for node_id in leaf_ind[0]]
ax.set_xticks(np.arange(len(leaf_ind[0])))
ax.set_xticklabels(xtick_labels)

if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_heatmap_single.pdf', 
            bbox_inches='tight')
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_heatmap_single.png', 
            bbox_inches='tight')
plt.show()

plant_ids = ['Marble River', 'Noble Clinton', 'Noble Ellenburg',
             'Noble Altona', 'Noble Chateaugay', 'Jericho Rise', 'Bull Run II Wind', 'Bull Run Wind']
feat_names = plant_list + ['Weather', 'Bias']

min_lag = [1, 4, 8, 16]
horizon_model_dict = {}
weight_mat_dict = {}

n_rows, n_cols = 4, 1
fig_width = 3.5
fig_height = 2.1*(n_rows / n_cols)  # Preserve aspect ratio
# fig_height = 3.5

# Titles
h_values = [1, 4, 8, 16]

weights_flattened = []
# Find weight matrices per horizon
for m in min_lag:
    with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{m}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
        FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
    
    horizon_model_dict[m] = FA_LEARN_LDR_LR_models_dict[10]
    
    target_model = horizon_model_dict[m]
    
    leaf_ind = np.where(np.array(target_model.feature)==-1)
    n_feat = len(target_model.target_features[0]) + len(target_model.fixed_features[0])
    
    weight_mat_dict[m] = np.zeros((len(leaf_ind[0]), 26))
    

    for i, leaf in enumerate(leaf_ind[0]):
        
        ### Coefficients
        w_opt = target_model.node_model_[leaf].model[0].weight.detach().numpy().reshape(-1)
        bias_opt = target_model.node_model_[leaf].model[0].bias.detach().numpy().reshape(-1)
    
        weight_mat_dict[m][i,:-1] = w_opt
        weight_mat_dict[m][i,-1] = bias_opt
    
    weights_flattened.append(weight_mat_dict[m].flatten())
    
    
weights_flattened = np.array(weights_flattened)

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, constrained_layout = True, 
                         figsize=(fig_width, fig_height), sharex=False, sharey=True)

# Flatten axes for easy iteration
axes = axes.flatten()

# Display heatmaps
vmin = min(h.min() for h in weight_mat_dict.values())
vmax = max(h.max() for h in weight_mat_dict.values())
v_abs_max = np.maximum(np.abs(vmin), np.abs(vmax))

for i, ax in enumerate(axes):
    
    print(f'Horizon:{h_values[i]}')
    
    plt.sca(axes[i])
    
    im = ax.imshow(weight_mat_dict[h_values[i]].T[::-1], aspect='auto', cmap='PiYG', 
                   interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)
    ax.set_title(rf"$h={h_values[i]}$", fontsize = 8)
    ax.tick_params(axis='both')
    
    # Find model in specific horizon
    temp_target_model = horizon_model_dict[h_values[i]]
    
    # Leaf indices for model
    leaf_ind = np.where(np.array(temp_target_model.feature)==-1)[0]

    # Annotate split feature
    for j, leaf in enumerate(leaf_ind):
        
        # missing pattern    
        missing_pattern = temp_target_model.missing_pattern[leaf]
        miss_ind = np.where(missing_pattern==1)[0]

        # plot fixed patterns 
        fixed_ind = temp_target_model.fixed_features[leaf]
        fixed_ind = np.array([s for s in fixed_ind if s not in list(miss_ind)] + [25])

        plt.plot( len(fixed_ind)*[j], 25-fixed_ind, '+', markersize = 5, 
                 color='black')
            
        if (missing_pattern==1).sum() == 0: 
            continue
        else: 
            plt.plot( len(miss_ind)*[j], 25-miss_ind, '_', markersize = 5, 
                     color='black')

    y_tick_ind = list(range(1,25,3))+[25, 26]
    y_tick_ind = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
    y_tick_ind = [24, 21, 18, 15, 12, 9, 6, 3, 1, 0]

    plt.yticks(y_tick_ind, feat_names, fontsize = 7)

    xtick_labels = [r'$\mathcal{U}_{%d}$' % node_id for node_id in leaf_ind]
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(xtick_labels, fontsize = 7)    


# Common colorbar
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.025, pad=0.02)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Magnitude", fontsize=7)


if config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_heatmap_all_horizons.pdf', 
            bbox_inches='tight')
    
plt.show()

