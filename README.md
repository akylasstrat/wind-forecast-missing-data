# Learning Data-Driven Uncertainty Set Partitions for Robust and Adaptive Energy Forecasting with Missing Data

This repository contains the code to recreate the experiments of

```
@misc{stratigakos2025learningdatadrivenuncertaintyset,
      title={Learning Data-Driven Uncertainty Set Partitions for Robust and Adaptive Energy Forecasting with Missing Data}, 
      author={Akylas Stratigakos and Panagiotis Andrianesis},
      year={2025},
      eprint={2503.20410},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2503.20410}, }
```

which is available [here](https://arxiv.org/abs/2503.20410).

### Abstract

Short-term forecasting models typically assume the availability of input data (features) when they are deployed and
in use. However, equipment failures, disruptions, cyberattacks, may lead to missing features when such models are used operationally, 
compromising forecast accuracy, and resulting in suboptimal operational decisions. 
In this work, we use adaptive robust optimization and adversarial machine learning to develop forecasting models that seamlessly handle missing
data operationally. 
We propose linear- and neural network-based forecasting models with parameters that adapt to available features, combining linear adaptation with a novel algorithm for learning data-driven uncertainty set partitions. The proposed
adaptive models do not rely on identifying historical missing data patterns and are suitable for real-time operations under
stringent time constraints. 
Extensive numerical experiments on short-term wind power forecasting considering horizons from 15
minutes to 4 hours ahead illustrate that our proposed adaptive models are on par with imputation when data are missing for
very short periods (e.g., when only the latest measurement is missing) whereas they significantly outperform imputation when
data are missing for longer periods. 
We further provide insights by showcasing how linear adaptation and data-driven partitions
(even with a few subsets) approach the performance of the optimal, yet impractical, method of retraining for every possible realization of missing data.

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
- ```\data```: Wind power data from the NYISO system, provided by NREL [here](https://research-hub.nrel.gov/en/publications/solar-wind-and-load-forecasting-dataset-for-miso-nyiso-and-spp-ba) (used under CC BY 3.0 US).
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
- **Generate plots**: Use ```create_plots.py``` to generate Figures 4-7.

For MacOS/Linux you need to update the project directory.
The results in .csv format provided in this repository are the ones appearing in the paper. Re-running all the experiments using the default parameters should save new .csv files with the same values as the ones provided here.


### Set-up

This code has been developed using ```Python```, version ```3.10.15```. To install the necessary packages, create a virtual environment using the ```conda create -n ENV_NAME python=3.10 ipython```, where ```ENV_NAME``` is the name of the environment.
Installt the necessary packages using ```pip install -r requirements.txt```.

Contact details: ```a.stratigakos@imperial.ac.uk```.

### Citation
Please use the reference provided above.
