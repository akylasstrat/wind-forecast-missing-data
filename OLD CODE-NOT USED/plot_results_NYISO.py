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

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)-LS}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)-LS}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(learn)^{10}-LS}$', 
                    'FA-lin-greedy-LS-1':'$\mathtt{FLA(learn)^{1}-LS}$', 
                    'FA-lin-greedy-LS-5':'$\mathtt{FLA(learn)^{5}-LS}$', 
                    'FA-lin-greedy-LS-2':'$\mathtt{FLA(learn)^{2}-LS}$', 
                    'FA-lin-greedy-LS-20':'$\mathtt{FLA(learn)^{20}-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(learn)^{10}-LS}$', 
                    'FA-fixed-NN':'$\mathtt{FA(fixed)-NN}$', 
                    'FA-greedy-NN':'$\mathtt{FA(learn)^{10}-NN}$', 
                    'FA-lin-fixed-NN':'$\mathtt{FLA(fixed)-NN}$', 
                    'FA-lin-greedy-NN':'$\mathtt{FLA(learn)^{10}-NN}$',
                    'v2FA-lin-fixed-NN':'$\mathtt{v2FA(fixed)-NN}$',
                    'v3FA-lin-fixed-NN':'$\mathtt{v3FA(fixed)-NN}$',
                    'NN':'$\mathtt{Imp-NN}$'}


models_to_common_labels = {'LS':'$\mathtt{Imp}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(learn)^{10}}$', 
                    'FA-lin-greedy-LS-1':'$\mathtt{FLA(learn)^{1}}$', 
                    'FA-lin-greedy-LS-5':'$\mathtt{FLA(learn)^{5}}$', 
                    'FA-lin-greedy-LS-2':'$\mathtt{FLA(learn)^{2}}$', 
                    'FA-lin-greedy-LS-20':'$\mathtt{FLA(learn)^{20}}$', 
                    'FA-greedy-LS':'$\mathtt{FA(learn)^{10}}$', 
                    'FA-fixed-NN':'$\mathtt{FA(fixed)}$', 
                    'FA-greedy-NN':'$\mathtt{FA(learn)^{10}}$', 
                    'FA-lin-fixed-NN':'$\mathtt{FLA(fixed)}$', 
                    'FA-lin-greedy-NN':'$\mathtt{FLA(learn)^{10}}$',
                    'NN':'$\mathtt{Imp}$'}

#%% Load data at turbine level, aggregate to park level
config = params()

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
weather_feat = True
weather_all_steps = True

config['min_lag'] = 1
freq = '15min'
nyiso_plants = ['Dutch Hill - Cohocton', 'Marsh Hill', 'Howard', 'Noble Clinton']
target_park = 'Noble Clinton'
config['save'] = True
min_lag = config['min_lag']
#%% Missing Data Completely at Random (MCAR)

### Load data for all forecast horizons
all_rmse = []
steps_ = [1, 4, 8, 16, 24]
for s in steps_:
    if (weather_all_steps == False):
        if s >= 8:
            temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
        else:
            temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results.csv', index_col = 0)            
    elif (weather_all_steps == True):        
        temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
        
    temp_df['steps'] = s
    
    all_rmse.append(temp_df)


all_rmse = pd.concat(all_rmse)

print('RMSE without missing data')
print((100*all_rmse.query(f'percentage==0').groupby(['steps']).mean()).round(2))

if weather_all_steps:
    all_rmse.to_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_all_RMSE_results_weather.csv')
else:
    all_rmse.to_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_all_RMSE_results.csv')
    
ls_models = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10','FA-lin-greedy-LS-20']
nn_models = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

#%%
(100*all_rmse.query('percentage>0.001').groupby(['steps']).mean()).round(2).to_clipboard()

#%% MCAR plots
LS_models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']
NN_models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

#### RMSE vs forecast horizon, both LS & NN, specific probability
nominal_rmse_LS = (100*all_rmse.query(f'percentage==0').groupby(['steps'])['LS'].mean()).round(2)

ave_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).mean())
std_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).std())

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']

print('Improvement over Imputation')
print( (100*((ave_improve_horizon['LS']-ave_improve_horizon['FA-lin-greedy-LS-10'])/ave_improve_horizon['LS'])).round(2) )

props = dict(boxstyle='round', facecolor='white', alpha=0.5)


fig, axes = plt.subplots(constrained_layout = True, nrows = 2, sharex = True, 
                         sharey = True, figsize = (3.5, 2.8))
plt.sca(axes[0])
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for i,m in enumerate(LS_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i])

# BASE CASE MODEL (no missing data)
for i, s in enumerate(steps_):
    plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_LS.loc[s]], '--', color = 'black')
    
plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)

nominal_rmse_NN = (100*all_rmse.query(f'percentage==0').groupby(['steps'])['NN'].mean()).round(2)

plt.sca(axes[1])
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
for i,m in enumerate(NN_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                 label = models_to_common_labels[m])

# BASE CASE MODEL (no missing data)    
for i, s in enumerate(steps_):
    # if i ==0:
    #     plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_NN.loc[s]], '--', color = 'black', label = '$\mathtt{NN}$')
    # else:
    plt.plot( np.arange(i, 5*0.1 + i, 0.1), 5*[nominal_rmse_NN.loc[s]], '--', color = 'black')

plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')

# Text to indicate forecasting model for each subplot
axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LS}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

axes[1].text(0.05, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.1))

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_LS_NN_abs_RMSE_vs_horizon_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_LS_NN_abs_RMSE_vs_horizon.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

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
    y_val[0] = 100*temp_df.query('percentage==0')['LS'].mean()

    plt.plot(x_val, y_val, color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.plot(x_val, len(x_val)*[100*all_rmse.query(f'percentage == 0 and steps == {min_lag}')['LS'].mean()], linestyle = '--', linewidth = 2, 
         color = 'black')

plt.ylabel('RMSE (%)')

# place a text box in upper left in axes coords
axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LS}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

plt.sca(axes[1])
for i, m in enumerate(NN_models_to_plot):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['NN'].mean()
    
    plt.plot(x_val, y_val, 
             label = models_to_common_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
# BASE case (no missing data)
plt.plot(x_val, len(x_val)*[100*all_rmse.query(f'percentage == 0 and steps == {min_lag}')['FA-fixed-NN'].mean()], linestyle = '--', linewidth = 2, 
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
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_joint_RMSE_MCAR_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_joint_RMSE_MCAR.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
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
    y_val[0] = 100*temp_df.query('percentage==0')['LS'].mean()
    
    plt.plot(x_val, y_val, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=1, fontsize = 6)

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_RMSE_MCAR_weather.pdf')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_RMSE_MCAR.pdf')
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
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_NN_RMSE_MCAR.pdf')
plt.show()
#%%
### NN: Check linear decision rules in input vs output layer
min_lag = 1
temp_df = all_rmse.query(f'(percentage in [0, 0.01, 0.05, 0.1]) and (steps == {min_lag})')
std_bar = 100*(temp_df.groupby(['percentage']).std())
x_val = temp_df['percentage'].unique().astype(float)

fig, ax = plt.subplots(constrained_layout = True)
for i, m in enumerate(['FA-lin-fixed-NN', 'v2FA-lin-fixed-NN', 'v3FA-lin-fixed-NN']):    
    y_val = 100*temp_df.groupby(['percentage'])[m].mean().values
    y_val[0] = 100*temp_df.query('percentage==0')['NN'].mean()
    plt.plot(x_val, y_val, 
             label = models_to_labels[m], color = colors[i], marker = marker[i], linestyle = '-', linewidth = 1)
    plt.fill_between(x_val, y_val- std_bar[m], y_val+ std_bar[m], alpha = 0.2, color = colors[i])    
    
plt.legend()
plt.ylabel('RMSE (%)')
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
plt.legend(ncol=1, fontsize = 6, loc = 'upper left')
plt.show()
stop_
#%%
# percentage improvement // LS
print((100* (temp_df.groupby(['percentage'])[['LS']].mean().values - temp_df.groupby(['percentage'])[LS_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2))
print( (100*temp_df.groupby(['percentage'])[LS_models_to_plot].mean()).round(2))

(100*((temp_df.groupby(['percentage'])[['LS']].mean().values - temp_df.groupby(['percentage'])[LS_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2)).to_clipboard()
(100*temp_df.groupby(['percentage'])[LS_models_to_plot].mean()).round(2).to_clipboard()

# percentage improvement // NN
print((100* (temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[NN_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2))
print( (100*temp_df.groupby(['percentage'])[NN_models_to_plot].mean()).round(2))

(100*((temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[NN_models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2)).to_clipboard()
(100*temp_df.groupby(['percentage'])[NN_models_to_plot].mean()).round(2).to_clipboard()


#%% Sensitivity analysis // LS

if weather_all_steps or (weather_feat and min_lag>=8):
    with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict_weather.pickle', 'rb') as handle:
        FA_lin_greedy_LS_models_dict = pickle.load(handle)           
else:
    with open(f'{cd}\\trained-models\\NYISO\\{freq}_{min_lag}_steps\\{target_park}_FA_lin_greedy_LS_models_dict.pickle', 'rb') as handle:
        FA_lin_greedy_LS_models_dict = pickle.load(handle)

### Find WC performance gap
WC_gap = []
for q in FA_lin_greedy_LS_models_dict.keys():
    temp_model = FA_lin_greedy_LS_models_dict[q]
    leaf_ind = np.where(np.array(temp_model.feature) == -1)[0]
        
    WC_gap.append(np.array(temp_model.Loss_gap_perc)[leaf_ind].max())

### Sensitivity analysis table results
models_to_plot = ['LS', 'FA-lin-fixed-LS','FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-2', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10',
                  'FA-lin-greedy-LS-20']

print( 100*all_rmse.query(f'percentage==0.1 and steps == {min_lag}').mean()[models_to_plot] )
print('WC Gap, percentage')
print((np.array(WC_gap)).round(2))

### Sensitivity plot
fig, ax1 = plt.subplots(constrained_layout = True, figsize = (3.5, 1.5))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

color = 'tab:red'
ax1.set_xlabel('Number of partitions $Q$')
ax1.set_ylabel('RMSE (%)')

temp_df = all_rmse.query(f'percentage==0.1 and steps == {min_lag}')
ax1.plot(5*[100*temp_df.mean()['LS']], color='black', marker = '2', label = '$\mathtt{Imp-LS}$', linewidth = 1)
ax1.plot(5*[100*temp_df.mean()['FA-lin-fixed-LS']], color='tab:brown', marker = 'd', label = '$\mathtt{FLA(fixed)}$', linewidth = 1)

ax1.plot(100*temp_df.mean()[models_to_plot[2:]].values, color='tab:green', marker = '8', label = '$\mathtt{FLA(learn)}^{Q}$', linewidth = 1)

ax1.tick_params(axis='y')
ax1.set_xticks(range(5), [1, 2, 5, 10, 20])

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.05))

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
# ax2.set_ylabel(r'$100\times\frac{\mathtt{UB-LB}}{\mathtt{LB}}$ (%)', color=color)  # we already handled the x-label with ax1
ax2.set_ylabel(r'$100\times\frac{\mathtt{UB-LB}}{\mathtt{LB}}$ (%)', color=color)  # we already handled the x-label with ax1

ax2.plot((np.array(WC_gap)).round(2), '-.', color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_sensitivity.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

#%% Missing Not at Random plots // all horizons

mnar_rmse_df = []

### Load results
for s in steps_:
    if (weather_all_steps == False):
        if s >= 8:
            temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
        else:
            temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{s}_steps_RMSE_results.csv', index_col = 0)            
    elif (weather_all_steps == True):        
        temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
        
    temp_df['steps'] = s
    
    mnar_rmse_df.append(temp_df)

mnar_rmse_df = pd.concat(mnar_rmse_df)

#### Barplot for specific forecast horizon
selected_step = 1
temp_mnar_df = mnar_rmse_df.query(f'steps == {selected_step}')
ls_models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']
nn_models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

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
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{selected_step}_MNAR_weather.pdf')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_{selected_step}_MNAR.pdf')
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
axes[0].text(0.05, 0.95, 'Forecasting model: $\mathtt{LS}$', transform=axes[0].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

axes[1].text(0.05, 0.95, 'Forecasting model: $\mathtt{NN}$', transform=axes[1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

lgd = fig.legend(fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.15, -0.1))

if weather_all_steps and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_LS_NN_MNAR_vs_horizon_weather.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
elif (weather_all_steps == False) and config['save']:
    plt.savefig(f'{cd}//plots//{freq}_{target_park}_LS_NN_MNAR_RMSE_vs_horizon.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


#%%
### scaled RMSE: divide by nominal model without missing data
scaled_rmse = all_rmse.query(f'percentage>0').copy()

for s in steps_:
    nominal_rmse_LS = all_rmse.query(f'steps == {s} and percentage==0')['LS'].mean()
    nominal_rmse_NN = all_rmse.query(f'steps == {s} and percentage==0')['NN'].mean()
    
    for m in ls_models:
        scaled_rmse.loc[scaled_rmse['steps'] == s, m] = scaled_rmse.loc[scaled_rmse['steps'] == s, m]/nominal_rmse_LS
    for m in nn_models:
        scaled_rmse.loc[scaled_rmse['steps'] == s, m] = scaled_rmse.loc[scaled_rmse['steps'] == s, m]/nominal_rmse_NN
        
### Percentage improvement over horizons
ls_models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']
nn_models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

all_rmse_horizon = all_rmse.copy()

all_rmse_horizon_impr = pd.DataFrame(data = [], columns = ls_models_to_plot + nn_models_to_plot + ['steps','percentage'])
all_rmse_horizon_impr['steps'] = all_rmse_horizon['steps']
all_rmse_horizon_impr['percentage'] = all_rmse_horizon['percentage']
all_rmse_horizon_impr[ls_models_to_plot] = ((all_rmse_horizon['LS'].values.reshape(-1,1) - all_rmse_horizon[ls_models_to_plot])/all_rmse_horizon['LS'].values.reshape(-1,1))
all_rmse_horizon_impr[nn_models_to_plot] = ((all_rmse_horizon['NN'].values.reshape(-1,1) - all_rmse_horizon[nn_models_to_plot])/all_rmse_horizon['NN'].values.reshape(-1,1))

all_rmse_horizon_impr[ls_models_to_plot] = ((all_rmse_horizon['LS'].values.reshape(-1,1) - all_rmse_horizon[ls_models_to_plot])/all_rmse_horizon['LS'].values.reshape(-1,1))

### Scaled RMSE vs forecast horizon
ave_improve_horizon = (scaled_rmse.query(f'percentage==0.1').groupby(['steps']).mean())
std_improve_horizon = (scaled_rmse.query(f'percentage==0.1').groupby(['steps']).std())

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(constrained_layout = True)

for i,m in enumerate(ls_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                 label = models_to_labels[m])

plt.ylabel('Scaled RMSE')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_LS_scaled_RMSE_vs_horizon.pdf')
plt.show()

fig, ax = plt.subplots(constrained_layout = True)
for i,m in enumerate(nn_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.1, 
                 ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                 label = models_to_labels[m])

plt.ylabel('Scaled RMSE')
plt.xticks(np.arange(len(steps_))+0.25, steps_)
plt.xlabel(r'Forecast horizon $h$ (15 minutes)')
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_NN_scaled_RMSE_vs_horizon.pdf')
plt.show()