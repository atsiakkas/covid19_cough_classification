# COVID-19 classification of cough audio files

This project performs data cleansing and pre-processing of cough audio files and implements a deep learning model for their classification on whether the individual on each audio recording had COVID-19 or not.


## Project

https://github.com/atsiakkas/covid19_cough_classification<br/>
<br/>


## Contents

**Dataset**: Contains the raw and processed data files

**EDA & Pre-processing**: Contains code for performing exploratory data analysis and data pre-processing including:
  1. Conversion to .wav format
  2. Filtering cough_detected < 0.8.
  3. Filtering "symptomatic" and unlabelled.
  4. Downsampling to 16khz.
  5. Standardising to 10 seconds by padding/cropping.
  6. Augmenting the binary labels and saving into a single .npz file.
  7. Importing data into a tensorflow format.

**Experiments**: Defines:
  1. MelSpectrogram layer: transforms the audio data into spectrograms allowing the application of a Convolutional Neural Network.
  2. SpecAugment layer: used as an augmentation technique to reduce model overfitting and smooth optimisation process.
  3. train function: Defines main model (based on the ResNet50 architeture) and performs training and evaluation.


## Key references

He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. and Le, Q.V., 2019. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779.

Orlandic, L., Teijeiro, T. and Atienza, D., 2021. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. Scientific Data, 8(1), pp.1-10.

Sharma, N., Krishnan, P., Kumar, R., Ramoji, S., Chetupalli, S.R., Ghosh, P.K. and Ganapathy, S., 2020. Coswara--a database of breathing, cough, and voice sounds for COVID-19 diagnosis. arXiv preprint arXiv:2005.10548.
