# Gradients without Backpropagation
## Classes of Github
We have created six different classes, every pair has implemented a Neural Network which we train with forward gradient in one of them and with backpropagation in the other one.     
We train a Logistic Regression model, a Multilayer Neural Network model and a Convolutional Neural Network model.
## Datasets used
For the Logistic Regression model and the MultiLayer Neural Network model we use the dataset 'penguins' from tensorflow.datasets (https://www.tensorflow.org/datasets/catalog/penguins?hl=es-419). To separate the data into the train and test data, we use train_test_split from sklearn.model_selection and then we have converted the inputs to type float and the targets to type long.      
For the Convolutional Neural Network model we have used the KMNIST dataset because we needed a dataset of 2-dimensional features to apply a convoultional 2d layer, although we have only picked 500 samples for the training set and 100 for the test set for higher speed. We have transofrmed the data to tensors to work with them and have used the data loader to obtain an iterator to access to the samples.
## Installations
Here are the packages that we have used:    

*import torch*       
*import tensorflow_datasets as tfds*: to import the penguins dataset      
*from sklearn.model_selection import train_test_split*: to separate the dataset into train data and test data      

*import torchvision*        
*from torchvision import transforms as T*: to convert the KMNIST dataset to tensors       
*from matplotlib import pyplot as plt*: to plot the evolution of the loss, the accuracy and the misclassifications of the model      

*from torch import nn* : to define our models      
*from torch.nn import functional as F*: to use activation functions      

*import functorch as fc*         
*from functorch import jvp*: to compute the jacobian multiplied by a tengent vector         
*from functools import partial*: to compute the jacobian       

*import time*      to compute the execution time of the training
## Description of the method
The objective of our project is to train different Neural Network models without using backpropagation, for this we use the forward AD mode.         
In general what we do is:         
Initialize the model parameters and compute the initial loss         
Until the loss is lower than a threshold we:
<ul>
<li>Define a perturbation vector for each parameter taken as a multivariate random variable (such that their scalar components are independent and have zero mean and unit variance)</li>
<li>Compute the loss and the directional derivative of the loss at each parameter in direction v simultaneously and without having to compute ∇loss in the process (forward-mode autodiff)</li>
<li>Multiply the scalar directional derivative ∇loss(θ)·v with vector v and obtain g(θ), the forward gradient (where θ represents each one of the parameters)</li>
<li>Update the parameters by substracting g(θ) multiplied by the learning rate</li>
<li>Recalculate the loss</li>
</ul>  
