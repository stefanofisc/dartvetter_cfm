This folder contains the code to build and train-test a Convolutional Neural Network. 
The architecture of this model consists of two branches: feature extraction and classification. 
The feature extraction branch is responsible for features extraction from the global view. 
The feature tensor produced by the last convolutional block is subjected to flattening. Then, 
classification is executed through utilization of a multi-layer perceptron within the classification 
branch.

- Files
m1.py: code to build the network architecture;
train_test_cnn.py: code to train and/or test the model;
read_extracted_features.py: this file contains methods to load features vectors processed by the 
Forest Diffusion model.

