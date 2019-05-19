# Beehive-Audio-Classification
This project involves feature selection and analysis for standard machine learning classification of audio beehive samples into three non-overlapping categories which are bee buzzing, cricket chirping and ambient noise.

The four standard machine learning models used are K-nearest neighbors, Random Forest, Logistic Regression and Support Vector Machines using OneVsRest strategy. All the model training was achieved using different modules of the scikit-learn library.
Features were selected by applying various filter based, wrapper based and embedded methods.

The feature_extraction folder contains all the scripts for extracting features from a wav file, feature_selection folder contains all the filter, wrapper and embedded methods to select best features from the 34 features of the BUZZ2 training dataset, model_codes folder contain the scripts for the best performing machine learning models on all datasets, plots folder contains the analysis plots of various feature extraction and selection strategies and finally the Terminal_outputs folder contains the performance results as it appeared on the terminal. Some folders also have a folder_description file that describes the contents of that folder.
