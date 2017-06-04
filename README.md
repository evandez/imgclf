# Overview
An convolutional neural network with drop out built from scratch. Its purpose is to classify images into one of six hard-coded categories: airplane, butterfly, flower, piano, starfish, and watch. The network supports arbitrarily many convolutional/pooling/fully connected layers  by exposing a simple builder pattern. Clients may also set hyperparameters in this fashion.

# Setup
Ensure you have Java SE 8 installed on your machine.

# Training and Testing
Run `make` at the top level to compile the code and then `make test` to train and test the network. The network will output its train, tune, and test accuracy at every epoch.

# Known Issues
- Convolution layers do not support biases.
- There is currently a bug in the convolution layer's backpropagation logic, causing test accuracy on this dataset to cap at 65%. 