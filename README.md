# COVID-19 classification of cough audio files

This project performs data cleansing and pre-processing of cough audio files and implements a deep learning model for their classification on whether the individual on each audio recording had COVID-19 or not.


## Getting started

Create and activate virtual environment:
```
python3 -m venv [name_of_venv]
source [name_of_venv]/bin/activate
```

Clone repository:
```
git clone https://github.com/atsiakkas/rl_vfa.git
```

Install requirements:
```
cd rl_vfa
pip install -e .
```
<br/>



## Project

https://github.com/atsiakkas/covid19_cough_classification<br/>
<br/>


## Contents

**Dataset**: Contains the raw and processed data files

**EDA & Pre-processing**: Contains code for performing exploratory data analysis and data pre-processing including:


**function_approximators**: Defines the classes of the function approximation models and of the replay buffer: ParametricModel, NeuralNetwork, LinearModel, NonParametricModel, DecisionTree, RandomForest, ExtraTrees, GradientBoostingTrees, SupportVectorRegressor, KNeighboursRegressor, GaussianProcess, eGaussianProcess, OnlineGaussianProcess

**plots**: Scripts (jupyter notebooks) for producing the plots used in the report and saved plots.

**results**: Saved output of runs (csv files).

**train**: Scripts (jupyter notebooks and .py files) for training and evaluation.

**utils**: Defines the training and plotting utility functions.<br/>
<br/>


## Key references

Reinforcement Learning algorithms: Richard S. Sutton and Andrew G. Barto. 2018. Reinforcement learning: An introduction. MIT press.

Fitted-Q Iteration: Damien Ernst, Pierre Geurts, and Louis Wehenkel. 2005. Tree-based batch mode
reinforcement learning. Journal of Machine Learning Research 6 (2005).

Online Gaussian Process: Lehel Csató and Manfred Opper. 2002. Sparse on-line Gaussian processes. Neural
computation 14, 3 (2002), 641–668.

Deep Q-Network: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness,
Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg
Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen
King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. 2015.
Human-level control through deep reinforcement learning. Nature 518, 7540
(2015). https://doi.org/10.1038/nature14236
