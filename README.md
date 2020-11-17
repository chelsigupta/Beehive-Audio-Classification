# Beehive-Audio-Classification

This project involves Automation of Feature Selection and Generation of Optimal Feature Subsets for Beehive
Audio Sample Classification into four non-overlapping categories which are bee buzzing, cricket chirping, ambient noise and lawn mowing.

The Automation includes 4 methods - Random Forest Feature Importance method, ReliefF feature selection method, Recursive Feature Elimination and Sequential Feature Selection. The four standard machine learning models used are K-nearest neighbors, Random Forest, Logistic Regression and Support Vector Machines using OneVsRest strategy. All the model training was achieved using different modules of the scikit-learn library.Features were selected by applying various filter based, wrapper based and embedded methods.

The feature_extraction folder contains all the scripts for extracting features from a wav file, there are 2 different code of feature extraction process i.e pyAudio analysis and librosa feature extraction. feature_selection folder contains all the filter, wrapper and embedded methods to select best features from the 34 features of the training dataset, model_codes folder contain the scripts for the best performing machine learning models on all datasets, plots folder contains the analysis plots of various feature extraction and selection strategies and finally the Terminal_outputs folder contains the performance results as it appeared on the terminal. Some folders also have a folder_description file that describes the contents of that folder.

In the Feature selection folder individual code for all the feature selection methods can be found.
