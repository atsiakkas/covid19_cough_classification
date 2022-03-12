# COVID-19 classification of cough audio files

This project performs data cleansing and pre-processing of cough audio recordings and implements a convolutional neural network model for their classification on whether the individual on each audio recording had COVID-19 or not. The crowd-sourced Coughvid [1] and Coswara [2] datasets were used for training and evaluating the model. Data pre-processing steps include filtering, down-sampling, audio segmentation, standardisation, and conversion to spectrograms. The code for cough detection and segmentation was adapted from [3]. The librosa library [4] was used for audio processing. A convolutional neural network (CNN) based on the ResNet-50 architecture [5] with pre-trained weights and implemented was used for the main classification model. The TensorFlow [6] library was used for training and evaluating the model. Two custom layers were implemented. The MelSpectrogram layer was applied to the inputs and converted them to mel-spectrograms so that a CNN model could be used. The SpecAugment layer [7] was used as an augmentation technique to reduce model overfitting and smooth optimisation process.


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

[1] Orlandic, L., Teijeiro, T. and Atienza, D., 2021. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. Scientific Data, 8(1), pp.1-10.

[2] Sharma, N., Krishnan, P., Kumar, R., Ramoji, S., Chetupalli, S.R., Ghosh, P.K. and Ganapathy, S., 2020. Coswara--a database of breathing, cough, and voice sounds for COVID-19 diagnosis. arXiv preprint arXiv:2005.10548.

[3] Orlandic, L., Teijeiro, T. and Atienza, D., 2021. Source code. https://c4science.ch/diffusion/10770/

[4] McFee, B., Raffel, C., Liang, D., Ellis, D.P., McVicar, M., Battenberg, E. and Nieto, O., 2015, July. librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference (Vol. 8, pp. 18-25).

[5] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[6] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M. and Kudlur, M., 2016. {TensorFlow}: A System for {Large-Scale} Machine Learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16) (pp. 265-283).

[7] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. and Le, Q.V., 2019. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779.

