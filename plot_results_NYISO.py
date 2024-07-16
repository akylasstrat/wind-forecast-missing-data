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

models_to_labels = {'LS':'$\mathtt{Imp-LS}$', 
                    'FA-fixed-LS':'$\mathtt{FA(fixed)^{\gamma}-LS}$',
                    'FA-lin-fixed-LS':'$\mathtt{FLA(fixed)^{\gamma}-LS}$',
                    'FA-lin-greedy-LS-10':'$\mathtt{FLA(learn)^{10}-LS}$', 
                    'FA-lin-greedy-LS-1':'$\mathtt{FLA(learn)^{1}-LS}$', 
                    'FA-lin-greedy-LS-5':'$\mathtt{FLA(learn)^{5}-LS}$', 
                    'FA-lin-greedy-LS-2':'$\mathtt{FLA(learn)^{2}-LS}$', 
                    'FA-lin-greedy-LS-20':'$\mathtt{FLA(learn)^{20}-LS}$', 
                    'FA-greedy-LS':'$\mathtt{FA(learn)^{10}-LS}$', 
                    'FA-fixed-NN':'$\mathtt{FA(fixed)^{\gamma}-NN}$', 
                    'FA-greedy-NN':'$\mathtt{FA(learn)^{10}-NN}$', 
                    'FA-lin-fixed-NN':'$\mathtt{FLA(fixed)^{\gamma}-NN}$', 
                    'FA-lin-greedy-NN':'$\mathtt{FLA(learn)^{10}-NN}$','v2FA-lin-fixed-NN':'$\mathtt{v2FA(fixed)-NN}$',
                    'NN':'$\mathtt{Imp-NN}$'}

#%% Load data at turbine level, aggregate to park level
config = params()

# min_lag: last known value, which defines the lookahead horizon (min_lag == 2, 1-hour ahead predictions)
# max_lag: number of historical observations to include
weather_feat = True
config['min_lag'] = 24
freq = '15min'
nyiso_plants = ['Dutch Hill - Cohocton', 'Marsh Hill', 'Howard', 'Noble Clinton']
target_park = 'Noble Clinton'
config['save'] = False
min_lag = config['min_lag']
#%% No missing data, all horizons

all_rmse = []
steps_ = [8, 16, 24]
for s in steps_:
    if weather_feat and s >= 16:
        temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results_weather.csv', index_col = 0)
    else:
        temp_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{s}_steps_RMSE_results.csv', index_col = 0)
        
    temp_df['steps'] = s
    
    all_rmse.append(temp_df)

all_rmse = pd.concat(all_rmse)

ls_models = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10','FA-lin-greedy-LS-20']
nn_models = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

scaled_coef_rmse = all_rmse.query(f'percentage>0').copy()
#%%
for s in steps_:
    nominal_rmse_LS = all_rmse.query(f'steps == {s} and percentage==0')['LS'].mean()
    nominal_rmse_NN = all_rmse.query(f'steps == {s} and percentage==0')['NN'].mean()
    
    for m in ls_models:
        scaled_coef_rmse.loc[scaled_coef_rmse['steps'] == s, m] = 1- (scaled_coef_rmse.loc[scaled_coef_rmse['steps'] == s, m] - nominal_rmse_LS)/(all_rmse.query(f'steps == {s} and percentage>0')['LS'].values - nominal_rmse_LS)

    for m in nn_models:
        scaled_coef_rmse.loc[scaled_coef_rmse['steps'] == s, m] = 1- (scaled_coef_rmse.loc[scaled_coef_rmse['steps'] == s, m] - nominal_rmse_NN)/(all_rmse.query(f'steps == {s} and percentage>0')['NN'].values - nominal_rmse_NN)
        
#%% Percentage improvement over horizons

ls_models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-10']
nn_models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']

all_rmse_horizon = all_rmse.copy()

all_rmse_horizon_impr = pd.DataFrame(data = [], columns = ls_models_to_plot + nn_models_to_plot + ['steps','percentage'])
all_rmse_horizon_impr['steps'] = all_rmse_horizon['steps']
all_rmse_horizon_impr['percentage'] = all_rmse_horizon['percentage']
all_rmse_horizon_impr[ls_models_to_plot] = ((all_rmse_horizon['LS'].values.reshape(-1,1) - all_rmse_horizon[ls_models_to_plot])/all_rmse_horizon['LS'].values.reshape(-1,1))
all_rmse_horizon_impr[nn_models_to_plot] = ((all_rmse_horizon['NN'].values.reshape(-1,1) - all_rmse_horizon[nn_models_to_plot])/all_rmse_horizon['NN'].values.reshape(-1,1))

all_rmse_horizon_impr[ls_models_to_plot] = ((all_rmse_horizon['LS'].values.reshape(-1,1) - all_rmse_horizon[ls_models_to_plot])/all_rmse_horizon['LS'].values.reshape(-1,1))

#%%

ave_improve_horizon = 100*(all_rmse_horizon_impr.query(f'percentage==0.1').groupby(['steps']).mean())
std_improve_horizon = 100*(all_rmse_horizon_impr.query(f'percentage==0.1').groupby(['steps']).std())

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(constrained_layout = True)

for i,m in enumerate(ls_models_to_plot):
    if m == 'LS':
        plt.plot(ave_improve_horizon[m].values, linestyle = '--', color = 'black', label = models_to_labels[m])
    else:        
        plt.errorbar(np.arange(len(steps_)), 
                     ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                     label = models_to_labels[m])

plt.ylabel('RMSE improvement (%)')
plt.xticks(np.arange(len(steps_)), steps_)
plt.xlabel(r'Forecast horizon $h$')
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_perc_improvement_vs_horizon.pdf')
plt.show()

#%%

ave_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).mean())
std_improve_horizon = 100*(all_rmse.query(f'percentage==0.1').groupby(['steps']).std())

marker = ['2', 'o', 'd', '^', '8', '1', '+', 's', 'v', '*', '^', 'p', '3', '4']
colors = ['black', 'tab:blue', 'tab:brown', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(constrained_layout = True)

for i,m in enumerate(ls_models_to_plot):

    plt.errorbar(np.arange(len(steps_))+i*0.075, 
                 ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values, linestyle = '', marker = marker[i], color = colors[i], 
                 label = models_to_labels[m])

plt.ylabel('RMSE (%)')
plt.xticks(np.arange(len(steps_))+0.15, steps_)
plt.xlabel(r'Forecast horizon $h$')
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_abs_improvement_vs_horizon.pdf')
plt.show()

#%%
fig, ax = plt.subplots(constrained_layout = True)

for i,m in enumerate(nn_models_to_plot):
    if m == 'NN':
        plt.plot(ave_improve_horizon[m].values, linestyle = '--', color = 'black', label = models_to_labels[m])
    else:
        plt.plot(np.arange(len(steps_)), ave_improve_horizon[m].values, marker = marker[i], color = colors[i], label = models_to_labels[m], linestyle=' ')
        
        # plt.errorbar(np.arange(3), ave_improve_horizon[m].values, yerr=std_improve_horizon[m].values)

plt.ylabel('RMSE improvement (%)')
plt.xticks(np.arange(len(steps_)), steps_)
plt.xlabel(r'Forecast horizon $h$')
plt.legend(ncol=1, fontsize = 6)
plt.show()



#%% Missing Not at Random
if weather_feat and min_lag >= 16:
    mae_df_nmar = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_MAE_results_weather.csv', index_col = 0)
    rmse_df_nmar = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_RMSE_results_weather.csv', index_col = 0)

    mae_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results_weather.csv', index_col = 0)
    rmse_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results_weather.csv', index_col = 0)
else:
    mae_df_nmar = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_MAE_results.csv', index_col = 0)
    rmse_df_nmar = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MNAR_{min_lag}_steps_RMSE_results.csv', index_col = 0)

    mae_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_MAE_results.csv', index_col = 0)
    rmse_df = pd.read_csv(f'{cd}\\results\\{freq}_{target_park}_MCAR_{min_lag}_steps_RMSE_results.csv', index_col = 0)

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
        yerr = 100*rmse_df_nmar[ls_models_to_plot].std(), label = '$\mathtt{LS}$')

plt.xticks(np.arange(0, 5*0.25, 0.25), ['Imp-LS', 'FA(fixed)-LS', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'], rotation = 45)

plt.bar(np.arange(1.5, 1.5+5*0.25, 0.25), 100*rmse_df_nmar[nn_models_to_plot].mean(), width = 0.2,
        yerr = 100*rmse_df_nmar[nn_models_to_plot].std(), label = '$\mathtt{NN}$')

plt.xticks(np.concatenate((np.arange(0, 5*0.25, 0.25), np.arange(1.5, 1.5+5*0.25, 0.25))), 
           ['$\mathtt{Imp-LS}$', '$\mathtt{FA(fixed)^{\gamma}-LS}$', 'FLA(fixed)-LS', 'FA(greedy)-LS', 'FLA(greedy)-LS'] + ['Imp-NN', 'FA(fixed)-NN', 'FLA(fixed)-NN', 'FA(greedy)-NN', 'FLA(greedy)-NN'], rotation = 45)


plt.ylim([100*rmse_df_nmar[ls_models_to_plot + nn_models_to_plot].mean().min()*0.8, 100*rmse_df_nmar[ls_models_to_plot + nn_models_to_plot].mean().max()*1.05])
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
# ax.set_xlim( 1, 11 )

# ax.grid( 'off', axis='x' )
# ax.grid( 'off', axis='x', which='minor' )

# vertical alignment of xtick labels
# va = [ 0, -.05, 0, -.05, -.05, -.05 ]
# for t, y in zip( ax.get_xticklabels( ), va ):
#     t.set_y( y )

# ax.tick_params( axis='x', which='minor', direction='out', length=30 )
# ax.tick_params( axis='x', which='major', bottom='off', top='off' )

# ax.set_xticks( xticks )

# ax.set_xticks( xticks_minor, minor=True )
# ax.set_xticklabels( xlbls )
# ax.set_xlim( 1, 11 )

if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_MNAR.pdf')
plt.show()


#%% LS performance degradation
 
# color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
#          'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive', 'cyan', 'yellow']

models_to_plot = ['LS', 'FA-fixed-LS', 'FA-lin-fixed-LS', 'FA-greedy-LS', 'FA-lin-greedy-LS-1']




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
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_LS_RMSE.pdf')
plt.show()



#%%
# percentage improvement
print((100* (temp_df.groupby(['percentage'])[['LS']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2).to_clipboard() )

print( (100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard())

#%%
(100* (temp_df.groupby(['percentage'])[['LS']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['LS']].mean().values).round(2).to_clipboard() 
#%%
(100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard()
#%% LS - sensitivity
 

models_to_plot = ['LS', 'FA-lin-greedy-LS-1', 'FA-lin-greedy-LS-5', 'FA-lin-greedy-LS-10',
                  'FA-lin-greedy-LS-20']

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
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6)
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_sensitivity_RMSE.pdf')
plt.show()

#%%
# temp_df = rmse_df.query('percentage==0.1')


#%% NN performance degradation


models_to_plot = ['NN', 'FA-fixed-NN', 'FA-lin-fixed-NN', 'FA-greedy-NN', 'FA-lin-greedy-NN']



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
plt.xlabel(r'Probability $\mathbb{P}_{0 \rightarrow 1}$')
plt.xticks(np.array(x_val), (np.array(x_val)).round(2))
# plt.ylim([9.9, 12.75])
plt.legend(ncol=1, fontsize = 6, loc = 'upper left')
if config['save']: plt.savefig(f'{cd}//plots//{freq}_{target_park}_{min_lag}_steps_NN_RMSE.pdf')
plt.show()

#%%

# percentage improvement
print((100* (temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['NN']].mean().values).round(2).to_clipboard() )

print( (100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard())
#%%
(100* (temp_df.groupby(['percentage'])[['NN']].mean().values - temp_df.groupby(['percentage'])[models_to_plot].mean())/ temp_df.groupby(['percentage'])[['NN']].mean().values).round(2).to_clipboard()

#%%
(100*temp_df.groupby(['percentage'])[models_to_plot].mean()).round(2).to_clipboard()



