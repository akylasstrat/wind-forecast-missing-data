# -*- coding: utf-8 -*-
"""
Check updated results 

@author: a.stratigakos
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

# models_to_labels = {'LR':'$\mathtt{Imp-LS}$', 
#                     'FA-FIXED-LR':'$\mathtt{FA(fixed)-LS}$',
#                     'FA-FIXED-LDR-LR':'$\mathtt{FLA(fixed)-LS}$',
#                     'FA-LEARN-LDR-LR-10':'$\mathtt{FLA(learn)^{10}-LS}$', 
#                     'FA-LEARN-LDR-LR-1':'$\mathtt{FLA(learn)^{1}-LS}$', 
#                     'FA-LEARN-LDR-LR-5':'$\mathtt{FLA(learn)^{5}-LS}$', 
#                     'FA-LEARN-LDR-LR-2':'$\mathtt{FLA(learn)^{2}-LS}$', 
#                     'FA-LEARN-LDR-LR-20':'$\mathtt{FLA(learn)^{20}-LS}$', 
#                     'FA-LEARN-LR-10':'$\mathtt{FA(learn)^{10}-LS}$', 
#                     'FA-FIXED-NN':'$\mathtt{FA(fixed)-NN}$', 
#                     'FA-LEARN-NN-10':'$\mathtt{FA(learn)^{10}-NN}$', 
#                     'FA-FIXED-LDR-NN':'$\mathtt{FLA(fixed)-NN}$', 
#                     'FA-LEARN-LDR-NN-10':'$\mathtt{FLA(learn)^{10}-NN}$',
#                     'NN':'$\mathtt{Imp-NN}$'}


# models_to_common_labels = {'LR':'$\mathtt{Imp}$', 
#                     'FA-FIXED-LR':'$\mathtt{FA(fixed)}$',
#                     'FA-FIXED-LDR-LR':'$\mathtt{FLA(fixed)}$',
#                     'FA-LEARN-LDR-LR-10':'$\mathtt{FLA(learn)^{10}}$', 
#                     'FA-LEARN-LDR-LR-1':'$\mathtt{FLA(learn)^{1}}$', 
#                     'FA-LEARN-LDR-LR-5':'$\mathtt{FLA(learn)^{5}}$', 
#                     'FA-LEARN-LDR-LR-2':'$\mathtt{FLA(learn)^{2}}$', 
#                     'FA-LEARN-LDR-LR-20':'$\mathtt{FLA(learn)^{20}}$', 
#                     'FA-LEARN-LR-10':'$\mathtt{FA(learn)^{10}}$', 
#                     'FA-FIXED-NN':'$\mathtt{FA(fixed)}$', 
#                     'FA-LEARN-NN-10':'$\mathtt{FA(learn)^{10}}$', 
#                     'FA-FIXED-LDR-NN':'$\mathtt{FLA(fixed)}$', 
#                     'FA-LEARN-LDR-NN-10':'$\mathtt{FLA(learn)^{10}}$',
#                     'NN':'$\mathtt{Imp}$'}

models_to_labels = {'LR':'$\mathtt{LR-Imp}$', 
                    'FA-FIXED-LR':'$\mathtt{LR-RF(fixed)}$',
                    'FA-FIXED-LDR-LR':'$\mathtt{LR-ARF(fixed)}$',
                    'FA-LEARN-LDR-LR-10':'$\mathtt{LR-ARF(learn)^{10}}$', 
                    'FA-LEARN-LDR-LR-1':'$\mathtt{LR-ARF(learn)^{1}}$', 
                    'FA-LEARN-LDR-LR-5':'$\mathtt{LR-ARF(learn)^{5}}$', 
                    'FA-LEARN-LDR-LR-2':'$\mathtt{LR-ARF(learn)^{2}}$', 
                    'FA-LEARN-LDR-LR-20':'$\mathtt{LR-ARF(learn)^{20}}$', 
                    'FA-LEARN-LR-10':'$\mathtt{LR-RF(learn)^{10}}$', 
                    'FA-FIXED-NN':'$\mathtt{NN-RF(fixed)}$', 
                    'FA-LEARN-NN-10':'$\mathtt{NN-RF(learn)^{10}}$', 
                    'FA-FIXED-LDR-NN':'$\mathtt{NN-ARF(fixed)}$', 
                    'FA-LEARN-LDR-NN-10':'$\mathtt{NN-ARF(learn)^{10}}$',
                    'NN':'$\mathtt{NN-Imp}$'}


models_to_common_labels = {'LR':'$\mathtt{Imp}$', 
                    'FA-FIXED-LR':'$\mathtt{RF(fixed)}$',
                    'FA-FIXED-LDR-LR':'$\mathtt{ARF(fixed)}$',
                    'FA-LEARN-LDR-LR-10':'$\mathtt{ARF(learn)^{10}}$', 
                    'FA-LEARN-LDR-LR-1':'$\mathtt{ARF(learn)^{1}}$', 
                    'FA-LEARN-LDR-LR-5':'$\mathtt{ARF(learn)^{5}}$', 
                    'FA-LEARN-LDR-LR-2':'$\mathtt{ARF(learn)^{2}}$', 
                    'FA-LEARN-LDR-LR-20':'$\mathtt{ARF(learn)^{20}}$', 
                    'FA-LEARN-LR-10':'$\mathtt{RF(learn)^{10}}$', 
                    'FA-FIXED-NN':'$\mathtt{RF(fixed)}$', 
                    'FA-LEARN-NN-10':'$\mathtt{RF(learn)^{10}}$', 
                    'FA-FIXED-LDR-NN':'$\mathtt{ARF(fixed)}$', 
                    'FA-LEARN-LDR-NN-10':'$\mathtt{ARF(learn)^{10}}$',
                    'NN':'$\mathtt{Imp}$'}

marker_dict = {
    "LR": {"marker": "x", "color": "black", 'markeredgewidth':1, 'label':'$\mathtt{Imp}$'},
    "FA-FIXED-LR": {"marker": "s", "color": "black", "markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{RF(fixed)}$'},
    "FA-FIXED-LDR-LR": {"marker": "o","color": "black","markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{ARF(fixed)}$'},
    "FA-LEARN-LR-10": {"marker": "s", "color": "black", "markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{RF(learn)^{10}}$'},
    "FA-LEARN-LDR-LR-10": {"marker": "o","color": "black","markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{ARF(learn)^{10}}$'},
    "NN": {"marker": "x", "color": "black", 'markeredgewidth':1, 'label':'$\mathtt{Imp}$'},
    "FA-FIXED-NN": {"marker": "s", "color": "black", "markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{RF(fixed)}$'},
    "FA-FIXED-LDR-NN": {"marker": "o","color": "black","markerfacecolor": "black",'markeredgewidth':1, 'label':'$\mathtt{ARF(fixed)}$'},
    "FA-LEARN-NN-10": {"marker": "s", "color": "black", "markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{RF(learn)^{10}}$'},
    "FA-LEARN-LDR-NN-10": {"marker": "o","color": "black","markerfacecolor": "none",'markeredgewidth':1, 'label':'$\mathtt{ARF(learn)^{10}}$'}}

#%% Load data at turbine level, aggregate to park level
config = params()

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
# weather_all_steps = True
# horizon = 1
# min_lag = horizon

freq = '15min'
nyiso_plants = ['Dutch Hill - Cohocton', 'Marsh Hill', 'Howard', 'Noble Clinton']
target_park = 'Noble Clinton'
config['save'] = True

#%% Missing Data Completely at Random (MCAR)

### Load data for all forecast horizons
all_rmse = []
steps_ = [1, 4, 8, 16, 24]
dataset = 'updated'
for s in steps_:
    # if (weather_all_steps == True):            
    try:
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_{dataset}.csv', index_col = 0)
    except:
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_full.csv', index_col = 0)
        
    temp_df['steps'] = s    
    all_rmse.append(temp_df)

all_rmse = pd.concat(all_rmse)
print('RMSE without missing data')
print((100*all_rmse.query('P_0_1==0').groupby(['steps']).mean()).round(2))

all_rmse.to_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_all_RMSE_results_full.csv')
    
# ls_models = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10','FA-lin-greedy-LS-20']
# nn_models = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']
#%%
LR_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
NN_models_to_plot = ['NN', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN', 'FA-LEARN-NN-10', 'FA-LEARN-LDR-NN-10']

(all_rmse.groupby(['P_0_1', 'P_1_0', 'num_series', 'steps']).mean())[LR_models_to_plot + NN_models_to_plot].to_clipboard()
#%% RMSE vs probabilities, grid with subplots

# Select parameters for subplots
p_0_1_list = [0.05, 0.1, 0.2]
p_1_0_list = [1, 0.2, 0.1]
step_list = [1, 4, 8, 16]
base_model = 'LR'
delta_step = 0.2
markersize = 4.5
fontsize = 7

full_experiment_list = list(itertools.product(p_1_0_list, p_0_1_list))


(100*all_rmse.query('P_0_1>0.001 and num_series>=4').groupby(['steps', 'P_0_1', 'P_1_0', 'num_series'])[LR_models_to_plot+NN_models_to_plot].mean()).round(2).to_clipboard()

# dictionary for subplots
ax_lbl = np.arange(9).reshape(3,3)
props = dict(boxstyle='round', facecolor='white', alpha=0.3)
# axis ratio
gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

fig, ax = plt.subplot_mosaic(ax_lbl, constrained_layout = True, figsize = (5.5, 1.05*3), 
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
    text_title = rf'$P_{{0,1}}={p_0_1}$'+'\n'+rf'$P_{{1,1}}={1-p_1_0}$'  
    # current_axis.title.set_text(text_title)
    # current_axis.text(0.025, 0.97, text_title, transform = current_axis.transAxes, 
    #                   fontsize=fontsize, verticalalignment='top', bbox=props)
    current_axis.text(0.65, 0.35, text_title, transform = current_axis.transAxes, 
                      fontsize=fontsize, verticalalignment='top', bbox=props)
        
    # model_lists = ([[base_model], ['FA-FIXED-'+base_model, 'FA-LEARN-'+base_model+'-10'], 
    #                       ['FA-FIXED-LDR-'+base_model, 'FA-LEARN-LDR-'+base_model+'-10']])
    # for k, sublist in enumerate(model_lists):
    #     # Line plots
    #     for m in sublist:
    #         y_val = 100*temp_df.groupby(['steps'])[m].mean()
    #         plt.plot(np.arange(len(step_list))+k*delta_step, 
    #                      y_val_horizon[m].values, linestyle = '', **marker_dict[m], markersize = markersize)

    # for l, s in enumerate(step_list):
        
    #     nom_line = current_axis.plot( np.arange(l, len(model_lists)*delta_step + l, delta_step), 
    #                                  len(model_lists)*[nominal_rmse_horizon.loc[s][base_model]], 
    #              '--', color = 'black', markersize = markersize)
    #     nom_line[0].set_dashes([3,1])

    for k,m in enumerate(models_to_plot):
        # Line plots
        y_val = 100*temp_df.groupby(['steps'])[m].mean()
        x_val = np.arange(len(step_list))+k*delta_step
        plt.plot(x_val, y_val_horizon[m].values, 
                 linestyle = '', **marker_dict[m], markersize = markersize)
        
    for l, s in enumerate(step_list):
        nom_line = current_axis.plot( np.arange(l, 5*delta_step + l, delta_step), 5*[nominal_rmse_horizon.loc[s][base_model]], 
                 '--', color = 'black', markersize = markersize, 
                 label = rf'$\mathtt{{{base_model}}}$')
        nom_line[0].set_dashes([3,1])

    plt.xticks(np.arange(len(step_list))+0.25, step_list)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# fig.set_xticks(x_val, np.arange(5))
ysuplabel = fig.supylabel('RMSE (%)')
fig.supxlabel(r'Forecast Horizon $h$ (15 minutes)')
plt.ylim([1.5, 20.5])

label_list = [rf'$\mathtt{{{base_model}}}-$' + l for l in labels[:5]] + [rf'$\mathtt{{{base_model}}}$ (no missing data)']

lgd = fig.legend(lines[:6], label_list, fontsize=fontsize, ncol = 3, loc = (1, .8), 
                 bbox_to_anchor=(0.25, -0.1))

if config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{base_model}_RMSE_MCAR_mat_{dataset}.pdf',  
                bbox_extra_artists=(lgd,ysuplabel), bbox_inches='tight')
plt.show()

#%% RMSE vs probabilities, single farm missing 

import itertools

LR_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
NN_models_to_plot = ['NN', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN', 'FA-LEARN-NN-10', 'FA-LEARN-LDR-NN-10']

mnar_rmse_df = []
weather_all_steps = True
step_list = [1,4,8,16]
p_0_1 = 0.2
p_1_0 = 0.1
delta_step = 0.15

temp_df = all_rmse.query(f'P_0_1=={p_0_1} and P_1_0=={p_1_0} and num_series==1 and steps in {step_list}')

#### Plot for all the horizons
ave_mnar_rmse_horizon = 100*(temp_df.groupby(['steps']).mean())
std_mnar_rmse_horizon = (temp_df.groupby(['steps']).std())

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
                         sharey = True, figsize = (3.5, 2.2))

plt.sca(axes[0])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for i,m in enumerate(LR_models_to_plot):
    plt.plot(np.arange(len(step_list))+i*delta_step, 
                 ave_mnar_rmse_horizon.loc[step_list][m].values, 
                 linestyle = '', **marker_dict[m])

# BASE CASE MODEL (no missing data)
for i, s in enumerate(step_list):
    plt.plot( np.arange(i, 5*delta_step + i, delta_step), 5*[nominal_rmse_horizon.loc[s]['LR']], '--', color = 'black', 
             label = 'No missing data')
    
plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.ylim([1.5, 20.5])

plt.sca(axes[1])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
for i,m in enumerate(NN_models_to_plot):

    plt.plot(np.arange(len(step_list))+i*delta_step, 
                 ave_mnar_rmse_horizon.loc[step_list][m].values, 
                 linestyle = '', **marker_dict[m])
    
    # plt.errorbar(np.arange(len(steps_))+i*0.1, 
    #              ave_mnar_rmse_horizon[m].values, yerr=std_mnar_rmse_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
    #              label = models_to_common_labels[m])

# BASE CASE MODEL (no missing data)    
for i, s in enumerate(step_list):
    plt.plot( np.arange(i, 5*delta_step + i, delta_step), 5*[nominal_rmse_horizon.loc[s]['NN']], '--', color = 'black', 
             label = 'No missing data')

plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(step_list))+0.25, step_list)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')
plt.ylim([1.5, 20.5])

# Text to indicate forecasting model for each subplot
axes[0].text(0.02, 0.95, 'Forecasting model: $\mathtt{LR}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

axes[1].text(0.02, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

lgd = fig.legend(lines[:6], labels[:6], ncol = 3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.125), fontsize = 6)


if config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_RMSE_single_farm_{dataset}.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%% Sensitivity plot// connected scatterplot
# load results

all_rmse = []
steps_ = [1]
min_lag = 1

for s in steps_:
    if (weather_all_steps == True):            
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_updated.csv', index_col = 0)
    temp_df['steps'] = s    
    all_rmse.append(temp_df)

all_rmse = pd.concat(all_rmse)
print('RMSE without missing data')
print((100*all_rmse.query('P_0_1==0').groupby(['steps']).mean()).round(2))

temp_df = all_rmse.query(f'P_0_1=={0.2} and P_1_0=={0.1} and num_series==8 and steps=={min_lag}')

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
Q_array = np.array(list(model_dictionary.keys()))
for q in Q_list:
    temp_model = model_dictionary[q]
    leaf_ind = np.where(np.array(temp_model.feature) == -1)[0]
        
    WC_gap.append(np.array(temp_model.Loss_gap_perc)[leaf_ind].max())
    WC_gap_dict[q] = np.array(temp_model.Loss_gap_perc)[leaf_ind].max()

    abs_gap.append(np.array(temp_model.Loss_gap)[leaf_ind].max())
    max_UB.append(np.array(temp_model.UB_Loss)[leaf_ind].max())
    best_LB.append(np.array(temp_model.LB_Loss)[leaf_ind].max())
    
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

plt.plot([1,2,5,10,20, 50], np.array(WC_gap), linestyle = '-', marker = '+')
plt.xticks([1,2,5,10,20, 50], [1,2,5,10,20, 50])
plt.ylabel('RMSE (%)')
plt.show()

### Sensitivity plot

Q_values_to_plot = [ 1,  2,  5, 10, 20, 50]
base_model = 'LR'
model_list = [f'FA-LEARN-LDR-{base_model}-{q}' for q in Q_values_to_plot]

fig, ax1 = plt.subplots(constrained_layout = True, figsize = (3.5, 1.5))

yval = 100*temp_df.mean()[model_list].values
xval = -WC_gap


text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'black', alpha=0.5)

plt.plot(xval, yval, marker = 'o', 
         color = 'black', markerfacecolor = 'white', markeredgewidth = 1, label = '$\mathtt{ARF(learn)^{10}}$', 
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
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


#%%
ax1.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='tab:green', marker = '8', label = '$\mathtt{LR-ARF(learn)}^{Q}$', linewidth = 1)

ax1.tick_params(axis='y')
ax1.set_xticks(range(5), [1, 2, 5, 10, 20])

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.1, -0.05))

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
# ax2.set_ylabel(r'$100\times\frac{\mathtt{UB-LB}}{\mathtt{LB}}$ (%)', color=color)  # we already handled the x-label with ax1
ax2.set_ylabel(r'Norm. Max. Gap (%)', color=color)  # we already handled the x-label with ax1

ax2.plot((np.array(WC_gap)).round(2), '-.', color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%% Interpreting performance/ subsection

target_model = FA_LEARN_LDR_LR_models_dict[10]

target_node = 0
n_feat = len(target_model.target_features[0]) + len(target_model.fixed_features[0])
### Coefficients
fig, axes = plt.subplots(constrained_layout = True, nrows = 1, sharex = True, 
                         sharey = False, figsize = (3.5, 2))

plt.bar(np.arange(n_feat)-0.2, target_model.node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1),
        width = 0.4, label = fr'$\omega^{{\text{{opt}}}}_{{{target_node}}}$')
plt.bar(np.arange(n_feat)+0.2, target_model.wc_node_model_[target_node].model[0].weight.detach().numpy().T.reshape(-1),
        width = 0.4, label = fr'$\omega^{{\text{{adv}}}}_{{{target_node}}}$')
plt.title(f'LR-ARF (node: {target_node})')
plt.ylabel('Coef. magnitude')
plt.xlabel('Features')
plt.legend()
plt.show()

#%%
### 
fig, axes = plt.subplots(constrained_layout = True, nrows = 1, sharex = True, 
                         sharey = False, figsize = (3.5, 1.5))

plt.plot( target_model.wc_node_model_[target_node].model[0].W[:,3].detach().numpy().T, label = r'$D^{\text{adv}}_{[3,:]}$')

plt.ylabel('Coef. magnitude')
plt.xlabel('Features')
plt.legend()
plt.show()


#%% Sensitivity analysis // LS
min_lag = 1
temp_df = all_rmse.query(f'P_0_1=={0.2} and P_1_0=={0} and num_series==8 and steps=={min_lag}')

with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LR_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LR_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_LDR_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_LDR_NN_models_dict = pickle.load(handle)           
with open(f'{cd}\\trained-models\\NYISO\\new_{freq}_{min_lag}_steps\\{target_park}_FA_LEARN_NN_models_dict_weather.pickle', 'rb') as handle:
    FA_LEARN_NN_models_dict = pickle.load(handle)                   

model_dictionary = FA_LEARN_LDR_LR_models_dict
### Find WC performance gap
WC_gap = []
abs_gap = []
max_UB = []
best_LB = []
Q_array = np.array(list(model_dictionary.keys()))
for q in np.sort(Q_array):
    temp_model = model_dictionary[q]
    leaf_ind = np.where(np.array(temp_model.feature) == -1)[0]
        
    WC_gap.append(np.array(temp_model.Loss_gap_perc)[leaf_ind].max())
    abs_gap.append(np.array(temp_model.Loss_gap)[leaf_ind].max())
    max_UB.append(np.array(temp_model.UB_Loss)[leaf_ind].max())
    best_LB.append(np.array(temp_model.LB_Loss)[leaf_ind].max())
### Sensitivity analysis table results
models_to_plot = ['LR', 'FA-FIXED-LDR-LR','FA-LEARN-LDR-LR-1', 'FA-LEARN-LDR-LR-2', 'FA-LEARN-LDR-LR-5', 
                  'FA-LEARN-LDR-LR-10', 'FA-LEARN-LDR-LR-20']

print( 100*all_rmse.query(f'P_0_1 == 0.2 and P_1_0 == 0.2 and steps == {min_lag}').mean()[models_to_plot] )
print('WC Gap, percentage')
print((np.array(WC_gap)).round(2))

text_props = dict(boxstyle='square', facecolor='white', edgecolor = 'tab:red', 
                  alpha=0.5)


### Absolute gap plot
fig, axes = plt.subplots(constrained_layout = True, nrows = 1, sharex = True, 
                         sharey = False, figsize = (3.5, 1.5))

plt.plot([1,2,5,10,20], np.array(WC_gap), linestyle = '-', marker = '+')
plt.xticks([1,2,5,10,20], [1,2,5,10,20])
plt.ylabel('RMSE (%)')
plt.show()

### Sensitivity plot
fig, ax1 = plt.subplots(constrained_layout = True, figsize = (3.5, 1.5))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

color = 'tab:red'
ax1.set_xlabel('Number of partitions $Q$')
ax1.set_ylabel('RMSE (%)')

ax1.plot(5*[100*temp_df.mean()['LR']], color='black', marker = '2', label = '$\mathtt{LR-Imp}$', linewidth = 1)
ax1.plot(5*[100*temp_df.mean()['FA-FIXED-LDR-LR']], color='tab:brown', marker = 'd', label = models_to_labels['FA-FIXED-LDR-LR'], linewidth = 1)

ax1.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='tab:green', marker = '8', label = '$\mathtt{LR-ARF(learn)}^{Q}$', linewidth = 1)

ax1.tick_params(axis='y')
ax1.set_xticks(range(5), [1, 2, 5, 10, 20])

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.1, -0.05))

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
# ax2.set_ylabel(r'$100\times\frac{\mathtt{UB-LB}}{\mathtt{LB}}$ (%)', color=color)  # we already handled the x-label with ax1
ax2.set_ylabel(r'Norm. Max. Gap (%)', color=color)  # we already handled the x-label with ax1

ax2.plot((np.array(WC_gap)).round(2), '-.', color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%%

# Lines
plt.plot(5*[100*temp_df.mean()['LR']],  color = 'black', linewidth = 1)

plt.plot(5*[100*temp_df.mean()['FA-FIXED-LDR-LR']], color = 'black', linewidth = 1)
plt.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='black', linewidth = 1)

# Markers
plt.plot(5*[100*temp_df.mean()['LR']], marker = marker_dict['LR']['marker'],  color = marker_dict['LR']['color'],
         markeredgewidth = marker_dict['LR']['markeredgewidth'], label = '$\mathtt{LR-Imp}$', linestyle = '')

plt.plot(5*[100*temp_df.mean()['FA-FIXED-LDR-LR']], marker = marker_dict['FA-FIXED-LDR-LR']['marker'],  color = marker_dict['FA-FIXED-LDR-LR']['color'],
         markeredgewidth = marker_dict['FA-FIXED-LDR-LR']['markeredgewidth'], 
         markerfacecolor = marker_dict['FA-FIXED-LDR-LR']['markerfacecolor'], 
         label = '$\mathtt{LR-ARF(fixed)}$', linestyle = '')

plt.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='black', 
         marker = marker_dict['FA-LEARN-LDR-LR-10']['marker'],
         markeredgewidth = marker_dict['FA-LEARN-LDR-LR-10']['markeredgewidth'], 
         markerfacecolor = 'white',
         label = '$\mathtt{LR-ARF(learn)}^{Q}$', linestyle = '')

plt.plot(5*[nominal_rmse_horizon.loc[min_lag]['LR']],  color = 'black', linewidth = 1, linestyle = '--' ,
             label = 'No missing data')

plt.ylabel('RMSE (%)')
plt.tick_params(axis='y')
plt.xticks(range(5), [1, 2, 5, 10, 20])

# Text to indicate forecasting model for each subplot
target_yval = 100*temp_df.mean()[models_to_plot[2:]].values
for i in range(len(WC_gap)):
    plt.text(i-0.15, .85*target_yval[i], f'${(np.array(WC_gap)).round(1)[i]}$%', fontsize=6,
            verticalalignment='top', bbox=text_props, color = 'tab:red')

lgd = plt.legend(fontsize=6, ncol=2, loc = (0.1, .5), 
                 bbox_to_anchor=(0.1, -.75))

plt.xticks(range(5), [1, 2, 5, 10, 20])
plt.xlabel('Number of subsets $Q$')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
    
plt.show()
#%%
### Sensitivity plot

fig, axes = plt.subplots(constrained_layout = True, nrows = 1, sharex = True, 
                         sharey = False, figsize = (3.5, 1.5))
# fig, ax1 = plt.subplots(constrained_layout = True, figsize = (3.5, 1.5))
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# color = 'tab:red'
# ax1.set_xlabel('Number of partitions $Q$')
# ax1.set_ylabel('RMSE (%)')

temp_df = all_rmse.query(f'P_0_1==0.1 and P_1_0 == 0.1 and steps == {min_lag} and num_series == 8')

# Lines
plt.plot(5*[100*temp_df.mean()['LR']],  color = 'black', linewidth = 1)

plt.plot(5*[100*temp_df.mean()['FA-FIXED-LDR-LR']], color = 'black', linewidth = 1)
plt.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='black', linewidth = 1)

# Markers
plt.plot(5*[100*temp_df.mean()['LR']], marker = marker_dict['LR']['marker'],  color = marker_dict['LR']['color'],
         markeredgewidth = marker_dict['LR']['markeredgewidth'], label = '$\mathtt{LR-Imp}$', linestyle = '')

plt.plot(5*[100*temp_df.mean()['FA-FIXED-LDR-LR']], marker = marker_dict['FA-FIXED-LDR-LR']['marker'],  color = marker_dict['FA-FIXED-LDR-LR']['color'],
         markeredgewidth = marker_dict['FA-FIXED-LDR-LR']['markeredgewidth'], 
         markerfacecolor = marker_dict['FA-FIXED-LDR-LR']['markerfacecolor'], 
         label = '$\mathtt{LR-ARF(fixed)}$', linestyle = '')

plt.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='black', 
         marker = marker_dict['FA-LEARN-LDR-LR-10']['marker'],
         markeredgewidth = marker_dict['FA-LEARN-LDR-LR-10']['markeredgewidth'], 
         markerfacecolor = 'white',
         label = '$\mathtt{LR-ARF(learn)}^{Q}$', linestyle = '')

plt.plot(5*[nominal_rmse_horizon.loc[min_lag]['LR']],  color = 'black', linewidth = 1, linestyle = '--' ,
             label = 'No missing data')

plt.ylabel('RMSE (%)')
plt.tick_params(axis='y')
plt.xticks(range(5), [1, 2, 5, 10, 20])

# (np.array(WC_gap)).round(2)

# Text to indicate forecasting model for each subplot
target_yval = 100*temp_df.mean()[models_to_plot[2:]].values
for i in range(len(WC_gap)):
    plt.text(i-0.15, .825*target_yval[i], f'${(np.array(WC_gap)).round(1)[i]}$%', fontsize=6,
            verticalalignment='top', bbox=text_props, color = 'tab:red')

lgd = plt.legend(fontsize=6, ncol=2, loc = (0.1, .5), 
                 bbox_to_anchor=(0.1, -.75))

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
# plt.ylim([1.5, 20.5])
# ax2.set_ylim([0, 75])

# color = 'tab:red'
# ax2.set_ylabel(r'$100\times\frac{\mathtt{UB-LB}}{\mathtt{LB}}$ (%)', color=color)  # we already handled the x-label with ax1

# axes[1].plot((np.array(WC_gap)).round(2), '-.', color=color, linewidth = 2)
# axes[1].tick_params(axis='y', labelcolor=color)
# axes[1].set_ylabel(r'Norm. Max. Gap (%)', color=color)  # we already handled the x-label with ax1


plt.xticks(range(5), [1, 2, 5, 10, 20])
plt.xlabel('Number of subsets $Q$')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
    
plt.show()

#%% Censored missing data// adversarial missingness mechanism

mnar_rmse_df = []
weather_all_steps = True
step_list = [1,4,8,16]
delta_step = 0.15
### Load results
for s in steps_:
    if (weather_all_steps) or (s>=8):
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_CENSOR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
    else:
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_CENSOR_{s}_steps_RMSE_results.csv', index_col = 0)
        
    temp_df['steps'] = s    
    mnar_rmse_df.append(temp_df)
mnar_rmse_df = pd.concat(mnar_rmse_df)

#### Plot for all the horizons
ave_mnar_rmse_horizon = 100*(mnar_rmse_df.groupby(['steps']).mean())
std_mnar_rmse_horizon = (mnar_rmse_df.groupby(['steps']).std())

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
                         sharey = True, figsize = (3.5, 2.2))

plt.sca(axes[0])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for i,m in enumerate(LR_models_to_plot):
    plt.plot(np.arange(len(step_list))+i*delta_step, 
                 ave_mnar_rmse_horizon.loc[step_list][m].values, 
                 linestyle = '', **marker_dict[m])

    # plt.errorbar(np.arange(len(step_list))+i*0.1, 
    #              ave_mnar_rmse_horizon[m].values, yerr=std_mnar_rmse_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i])

# BASE CASE MODEL (no missing data)
for i, s in enumerate(step_list):
    plt.plot( np.arange(i, 5*delta_step + i, delta_step), 5*[nominal_rmse_horizon.loc[s]['LR']], '--', color = 'black', 
             label = 'No missing data')
    
plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.ylim([1.5, 20.5])

plt.sca(axes[1])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
for i,m in enumerate(NN_models_to_plot):

    plt.plot(np.arange(len(step_list))+i*delta_step, 
                 ave_mnar_rmse_horizon.loc[step_list][m].values, 
                 linestyle = '', **marker_dict[m])
    
    # plt.errorbar(np.arange(len(steps_))+i*0.1, 
    #              ave_mnar_rmse_horizon[m].values, yerr=std_mnar_rmse_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
    #              label = models_to_common_labels[m])

# BASE CASE MODEL (no missing data)    
for i, s in enumerate(step_list):
    plt.plot( np.arange(i, 5*delta_step + i, delta_step), 5*[nominal_rmse_horizon.loc[s]['NN']], '--', color = 'black', 
             label = 'No missing data')

plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(step_list))+0.25, step_list)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')
plt.ylim([1.5, 20.5])

# Text to indicate forecasting model for each subplot
axes[0].text(0.02, 0.95, 'Forecasting model: $\mathtt{LR}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

axes[1].text(0.02, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

lgd = fig.legend(lines[:6], labels[:6], ncol = 3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.125), fontsize = 6)


# lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
#                  bbox_to_anchor=(0.15, -0.1))

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_CENSOR_vs_horizon_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_CENSOR_vs_horizon.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


#%% MCAR plots
# LS_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
# NN_models_to_plot = ['NN', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN', 'FA-LEARN-NN-10', 'FA-LEARN-LDR-NN-10']

# #### RMSE vs forecast horizon, both LS & NN, specific probability
# nominal_rmse_LS = (100*all_rmse.query(f'percentage==0').groupby(['steps'])['LR'].mean()).round(2)

# ave_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).mean())
# std_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).std())

# marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
# colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']

# print('Improvement over Imputation')
# print( (100*((ave_improve_horizon['LR']-ave_improve_horizon['FA-LEARN-LDR-LR-10'])/ave_improve_horizon['LR'])).round(2) )

# props = dict(boxstyle='round', facecolor='white', alpha=0.5)


# fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
#                          sharey = True, figsize = (3.5, 2.8))
# plt.sca(axes[0])
# axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# for i,m in enumerate(LS_models_to_plot):

#     plt.errorbar(np.arange(len(steps_))+i*0.1, 
#                  ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i])

# # BASE CASE MODEL (no missing data)
# for i, s in enumerate(steps_):
#     plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_LS.loc[s]], '--', color = 'black')
    
# plt.ylabel('RMSE (%)')
# plt.xticks(np.arange(len(steps_))+0.25, steps_)

# nominal_rmse_NN = (100*all_rmse.query(f'percentage==0').groupby(['steps'])['NN'].mean()).round(2)

# plt.sca(axes[1])
# axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# for i,m in enumerate(NN_models_to_plot):

#     plt.errorbar(np.arange(len(steps_))+i*0.1, 
#                  ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
#                  label = models_to_common_labels[m])

# # BASE CASE MODEL (no missing data)    
# for i, s in enumerate(steps_):
#     # if i ==0:
#     #     plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_NN.loc[s]], '--', color = 'black', label = '$\mathtt{NN}$')
#     # else:
#     plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_NN.loc[s]], '--', color = 'black')

# plt.ylabel('RMSE (%)')
# plt.xticks(np.arange(len(steps_))+0.25, steps_)
# plt.xlabel(r'Forecast horizon $h$ (15 minutes)')

# # Text to indicate forecasting model for each subplot
# axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LR}$', transform=axes[0].transAxes, fontsize=6,
#         verticalalignment='top', bbox=props)

# axes[1].text(0.05, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
#         verticalalignment='top', bbox=props)

# lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
#                  bbox_to_anchor=(0.15, -0.1))

# if weather_all_steps and config['save']:
#     plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_LS_NN_abs_RMSE_vs_horizon_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
# elif (weather_all_steps == False) and config['save']:
#     plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_LS_NN_abs_RMSE_vs_horizon.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()

#%% Detailed RMSE for a specific horizon, across all probabilities

### LS & NN performance degradation vs percentage of missing data
min_lag = 1

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']

ls_colors = plt.cm.tab20c( list(np.arange(3)))
lad_colors = plt.cm.tab20( list(np.arange(10,12)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])

# colors = ['black'] + list(ls_colors) +list(lad_colors) + list(fdr_colors) 

colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']
line_style = ['--' '-', '-', '-']


props = dict(boxstyle='round', facecolor='white', alpha=0.5)

fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
                         sharey = True, figsize = (3.5, 2.8))

plt.sca(axes[0])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

temp_df = all_rmse.query(f'(percentage in [0, 0.01, 0.05, 0.1]) and (steps == {min_lag})')
std_bar = 100*(temp_df.groupby(['percentage'])[LS_models_to_plot + NN_models_to_plot].std())
x_val = temp_df['percentage'].unique().astype(float)

for i, m in enumerate(LS_models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['LR'].mean()

    plt.plot(x_val, y_val, color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.plot(x_val, len(x_val)*[100*all_rmse.query(f'percentage == 0 and steps == {min_lag}')['LR'].mean()], linestyle = '--', linewidth = 2, 
         color = 'black')

plt.ylabel('RMSE (%)')

# place a text box in upper left in axes coords
axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LR}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

plt.sca(axes[1])
for i, m in enumerate(NN_models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['NN'].mean()
    
    plt.plot(x_val, y_val, 
             label = models_to_common_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
# BASE case (no missing data)
plt.plot(x_val, len(x_val)*[100*all_rmse.query(f'percentage == 0 and steps == {min_lag}')['FA-FIXED-NN'].mean()], linestyle = '--', linewidth = 2, 
         color = 'black',)

plt.ylabel('RMSE (%)')
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
# place a text box in upper left in axes coords
axes[1].text(0.05, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([1.5,13.5])

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.1))

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_joint_RMSE_MCAR_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_joint_RMSE_MCAR.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%% LS performance degradation vs Probability of missing data
 
min_lag = 1
fig, ax = plt.subplots(constrained_layout = True)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

temp_df = all_rmse.query(f'(percentage in [0, 0.01, 0.05, 0.1]) and (steps == {min_lag})')
std_bar = 100*(temp_df.groupby(['percentage']).std())
x_val = temp_df['percentage'].unique().astype(float)

for i, m in enumerate(LS_models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['LR'].mean()
    
    plt.plot(x_val, y_val, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=1, fontsize = 6)

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_RMSE_MCAR_weather.pdf')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_LS_RMSE_MCAR.pdf')
plt.show()

### NN performance degradation
fig, ax = plt.subplots(constrained_layout = True)
for i, m in enumerate(NN_models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['NN'].mean()
    plt.plot(x_val, y_val, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6, loc = 'upper left')
if config['save']: plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{min_lag}_steps_NN_RMSE_MCAR.pdf')
plt.show()

#%%
# percentage improvement // LS
print((100* (temp_df.groupby(['percentage'])[['LR']].mean().values - temp_df.groupby(['percentage'])[LS_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LR']].mean().values).round(2))
print( (100*temp_df.groupby(['percentage'])[LS_models_to_plot].mean()).round(2))

(100*((temp_df.groupby(['percentage'])[['LR']].mean().values - temp_df.groupby(['percentage'])[LS_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LR']].mean().values).round(2)).to_clipboard()
(100*temp_df.groupby(['percentage'])[LS_models_to_plot].mean()).round(2).to_clipboard()

# percentage improvement // NN
print((100* (temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[NN_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['NN']].mean().values).round(2))
print( (100*temp_df.groupby(['percentage'])[NN_models_to_plot].mean()).round(2))

(100*((temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[NN_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['NN']].mean().values).round(2)).to_clipboard()
(100*temp_df.groupby(['percentage'])[NN_models_to_plot].mean()).round(2).to_clipboard()




#%% Missing Not at Random plots // all horizons

mnar_rmse_df = []

### Load results
for s in steps_:
    if (weather_all_steps) or (s>=8):
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MNAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
    else:
        temp_df = pd.read_csv(f'{cd}\\new_results\\{freq}_{target_park}_MNAR_{s}_steps_RMSE_results.csv', index_col = 0)
        
    temp_df['steps'] = s    
    mnar_rmse_df.append(temp_df)
mnar_rmse_df = pd.concat(mnar_rmse_df)

#### Barplot for specific forecast horizon
selected_step = 1
temp_mnar_df = mnar_rmse_df.query(f'steps == {selected_step}')
ls_models_to_plot = ['LR', 'FA-FIXED-LR', 'FA-FIXED-LDR-LR', 'FA-LEARN-LR-10', 'FA-LEARN-LDR-LR-10']
nn_models_to_plot = ['NN', 'FA-FIXED-NN', 'FA-FIXED-LDR-NN', 'FA-LEARN-NN-10', 'FA-LEARN-LDR-NN-10']

fig, ax = plt.subplots(constrained_layout = True)
plt.bar(np.arange(0, 5*0.25, 0.25), 100*temp_mnar_df[ls_models_to_plot].mean(), width = 0.2, alpha = .3, 
        yerr = 100*temp_mnar_df[ls_models_to_plot].std(), label = '$\mathtt{LS}$')

plt.xticks(np.arange(0, 5*0.25, 0.25), ['Imp-LS', 'FA(fixed)-LS', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'], rotation = 45)

plt.bar(np.arange(1.5, 1.5+5*0.25, 0.25), 100*temp_mnar_df[nn_models_to_plot].mean(), width = 0.2,
        yerr = 100*temp_mnar_df[nn_models_to_plot].std(), label = '$\mathtt{NN}$')

plt.xticks(np.concatenate((np.arange(0, 5*0.25, 0.25), np.arange(1.5, 1.5+5*0.25, 0.25))), 
           ['$\mathtt{Imp-LS}$', '$\mathtt{FA(fixed)^{\gamma}-LS}$', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'] + ['Imp-NN', 'FA(fixed)-NN', 'FLA(fixed)-NN', 'FA(greedy)-NN', 'FLA(greedy)-NN'], rotation = 45)


plt.ylim([100*temp_mnar_df[ls_models_to_plot + nn_models_to_plot].mean().min()*0.8, 100*temp_mnar_df[ls_models_to_plot + nn_models_to_plot].mean().max()*1.1])
plt.ylabel("RMSE (%)")

xticks_minor = np.concatenate((np.arange(0, 5*0.25, 0.25), np.arange(1.5, 1.5+5*0.25, 0.25)))
xticks_major = [0.625 + 2]
xlbls = [ '$\mathtt{Imp}$', '$\mathtt{FA(fixed)^{\gamma}}$', '$\mathtt{FLA(fixed)^{\gamma}}$',
          '$\mathtt{FA(learn)^{10}}$', '$\mathtt{FLA(learn)^{10}}$'] + [ '$\mathtt{Imp}$', '$\mathtt{FA(fixed)^{\gamma}}$', '$\mathtt{FLA(fixed)^{\gamma}}$',
                    '$\mathtt{FA(learn)^{10}}$', '$\mathtt{FLA(learn)^{10}}$']


# ax.set_xticks( xticks_major )
ax.set_xticks( xticks_minor, minor=True )
ax.set_xticklabels( xlbls, rotation = 45, fontsize = 7)

plt.legend(ncol = 2)

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{selected_step}_MNAR_weather.pdf')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_{selected_step}_MNAR.pdf')
plt.show()


#### Plot for all the horizons
ave_mnar_rmse_horizon = 100*(mnar_rmse_df.groupby(['steps']).mean())
std_mnar_rmse_horizon = (mnar_rmse_df.groupby(['steps']).std())


props = dict(boxstyle='round', facecolor='white', alpha=0.5)
fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
                         sharey = True, figsize = (3.5, 2.8))

plt.sca(axes[0])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for i,m in enumerate(ls_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_mnar_rmse_horizon[m].values, yerr=std_mnar_rmse_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i])

# BASE CASE MODEL (no missing data)
for i, s in enumerate(steps_):
    plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_LS.loc[s]], '--', color = 'black')
    
plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)

nominal_rmse_NN = (100*all_rmse.query(f'percentage==0').groupby(['steps'])['NN'].mean()).round(2)

plt.sca(axes[1])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
for i,m in enumerate(nn_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_mnar_rmse_horizon[m].values, yerr=std_mnar_rmse_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                 label = models_to_common_labels[m])

# BASE CASE MODEL (no missing data)    
for i, s in enumerate(steps_):
    plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_NN.loc[s]], '--', color = 'black')

plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')

# Text to indicate forecasting model for each subplot
axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LR}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

axes[1].text(0.05, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.1))

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_LS_NN_MNAR_vs_horizon_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//new_plots//{freq}_{target_park}_LS_NN_MNAR_RMSE_vs_horizon.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


