# Adaptive Robust Optimization for Energy Forecasting with Missing Data

This repository contains the code to recreate the experiments of

```
@unpublished{xxxx,
TITLE = {{Adaptive Robust Optimization for Energy
Forecasting with Missing Data}}, AUTHOR = {Stratigakos, Akylas and Andrianesis, Panagiotis and Kariniotakis, Georges}}
```

which is available [here]().

### Abstract

Short-term energy forecasting is critical for the near real-time operation of low-carbon power systems.
Equipment failures, network latency, cyberattacks, 
may lead to missing feature data operationally, threatening forecast accuracy, 
resulting in suboptimal operational decisions, and driving system costs and risk.
In this work, we leverage adaptive robust optimization and adversarial machine learning to develop forecasting models whose parameters adapt to the \hl{available features, trained by minimizing the worst-case loss due to missing data}. Considering both linear and neural network-based forecasting models, 
we combine linear decision rules with a novel algorithm for learning data-driven uncertainty set partitions, enabling model parameters to be a linear, a piecewise constant, or a piecewise linear function of available features. 
Notably, our adaptation methods do not make assumptions about the missing data, do not require access to training data with incomplete observations, and are suitable for real-time operations under stringent time constraints. 
We examine the efficacy of our proposed adaptation methods in a comprehensive application of short-term wind power forecasting, using spatiotemporal data from adjacent wind power plants and considering forecast horizons from 15 minutes to 6 hours ahead.
The results show that our proposed adaptation methods significantly outperform the industry gold standard impute-then-regress benchmark, with an average improvement of over 20\% across all forecast horizons.

---

### Code Organization

Running the experiments:
- ```clean_main_model_train_NYISO_weather.py```: Trains all the forecasting models.
- ```clean_main_model_testing_NYISO_weather.py```: Runs the missing data experiments, saves the results.
- ```clean_plot_results_NYISO.py```: Summarizes results, plots Figures.

Other scripts:
- ```clean_torch_custom_layers.py```: Implementation of Robust Regression and Adaptive Robust Regression in PyTorch.
- ```clean_finite_adaptability_functions.py```: Implements uncertainty set partitioning.
- ```utility_functions.py```: Includes auxiliary functions.

Input/ output data and required packages:
- ```\data```: Wind power data from the NYISO system, provided by NREL [here](https://research-hub.nrel.gov/en/publications/solar-wind-and-load-forecasting-dataset-for-miso-nyiso-and-spp-ba).
```@incollection{nrel2024dataset,
  title={Solar PV, Wind Generation, and Load Forecasting Dataset for 2018 across MISO, NYISO, and SPP Balancing Areas}, author={Bryce, Richard and Feng, Cong and Sergi, Brian and Ring-Jarvi, Ross and Zhang, Wenqi and Hodge, Bri-Mathias}, year={2023}, publisher={Report, National Renewable Energy Laboratory}, doi = {NREL/TP-5D00-83828}, url={https://www.nrel.gov/docs/fy24osti/83828.pdf}}
```
- ```\trained-models```: Stores trained robust models for different forecast horizons.
- ```\results```: Stores experimental results for different forecast horizons.
- ```\plots```: Stores plotted figures (includes additional figures that do not appear in the published paper).
- ```requirements.txt```: Required packages and libraries.
---

### Reproducing the Results

To reproduce the results, the following steps are required:
- **Train models**: To train the forecasting models use  ```clean_main_model_train_NYISO_weather.py```.
Function ```params()``` sets the experimental setup (train/test split, selected wind farm, forecast horizon etc.).
To recreate the results, leave parameters at default values and change only ```params['horizon']``` to control the respective forecast horizon. 
Run the script once by setting ```params['horizon']``` to each value in ``` = [1, 4, 8, 16, 24]```.
By setting ```params['save'] = True``` (default value), new models will be stored in the respective subfolder in ```\trained-models```.
- **Missing data experiments**: Given trained models, run the ```clean_main_model_test_NYISO_weather.py``` to implement the missing data experiments. 
As before, use ```params['horizon']``` to control the forecast horizon and run the script once for each value in ``` = [1, 4, 8, 16, 24]```.
- **Generate plots**: Use ```clean_plot_results_NYISO.py``` to generate Figures 4-7.

For MacOS/Linux you need to update the project directory.
The results in .csv format provided in this repository are the ones appearing in the paper. Re-running all the experiments using the default parameters should save new .csv files with the same values as the ones provided here.


### Set-up

This code has been developed using ```Python```, version ```3.10.15```. To install the necessary packages, create a virtual environment using the ```conda create -n ENV_NAME python=3.10 ipython```, where ```ENV_NAME``` is the name of the environment.
Installt the necessary packages using ```pip install -r requirements.txt```.

Contact details: ```a.stratigakos@imperial.ac.uk```.

### Citation
Please use the reference provided above.
