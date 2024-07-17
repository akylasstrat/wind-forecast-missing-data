How to read file names:

{Frequency}_{Plant Name}_{Forecast Horizon}_{Base model}_{Error metric}_{Missing Data Mechanism}_{Does it include weather as feature}

Indicates performance degradation vs probability of missing data, when data are MCAR
Example: 15min_Noble Clinton_1_steps_NN_RMSE_MCAR
	- 15 min measurement frequency
	- Plant name: Noble Clinton
	- Horizon: 1 step ahead (15-min ahead)
	- Base model: NN
	- Data are MCAR
	- Metric: RMSE degradation vs probability of missing data