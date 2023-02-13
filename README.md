# Gradients without Backpropagation
## Classes of Github
We have created six different classes, every pair has implemented a Neural Network which we train with forward gradient in one of them and with backpropagation in the other one. We train a Logistic Regression model, a Multilayer Neural Network model and a Convolutional Neural Network model.
## Datasets used
For the Logistic Regreesion model and the MultiLayer Regression model we use the dataset 'penguins' from tensorflow.datasets (https://www.tensorflow.org/datasets/catalog/penguins?hl=es-419). To separate the data into the train and test data, we use train_test_split from sklearn.model_selection and then we have converted the inputs to type float and the targets to type long.
For the Convolutional Neural Network model we have used the KMNIST dataset because we needed a dataset of 2-dimensional features to apply a convoultional 2d layer, although we have only picked 500 samples for the training set and 100 for the test set for higher speed. We have transofrmed the data to tensors to work with them and have used the data loader to obtain an iterator to access to the samples.
## Installations
Here are the packages that we have used:

*import torch*
*import tensorflow_datasets as tfds*    to import the penguins dataset 
*from sklearn.model_selection import train_test_split*    to separate the dataset into train data and test data 

*import torchvision*
*from torchvision import transforms as T*   to convert the KMNIST dataset to tensors
*from matplotlib import pyplot as plt*    to plot the evolution of the loss, the accuracy and the misclassifications of the model

*from torch import nn*
*from torch.nn import functional as F*    to use activation functions

*import functorch as fc*
*from functorch import jvp*   to compute the jacobian multiplied by a tengent vector
*from functools import partial*   to compute the jacobian

*import time*   to compute the execution time of the training
## Description of the method
The objective of our project is to train different Neural Network models without using backpropagation, for this we use the forwar AD mode.
In general what we do is:
Initialize the model parameters and compute the initial loss
Until the loss is lower than a threshold we:

