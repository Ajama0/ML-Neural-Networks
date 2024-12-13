

import numpy as np
import matplotlib.pyplot as py
import random as random
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()




#X, y = spiral_data(100,3) #100 feature sets per class, each class has 100 feature sets = 300
                        #in total we have 300 features and 2 features per feature set (300,2)

np.random.seed(0)


class Layer_dense:


    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_layer:        

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_softmax:

    def Softmax(self, inputs):
        before_exp  = inputs - np.max(inputs, axis= 1, keepdims= True)
        exp_values = np.exp(before_exp)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims= True)


class Loss:
    def Calculate(self, output, y):
        sample_loss = self.forward(output, y) 
        date_loss = np.mean(sample_loss)
        return date_loss
    

class Loss_crossEntropy(Loss):
    def forward(self, y_pred, y_true):
        #we want to know how many input samples in our inputs
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1: #this means class targets are vectors
            correct_confindences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confindences = np.sum(np.multiply(y_pred,y_true), axis=1)

        negative_log = -np.log(correct_confindences)  
        return negative_log  
    
                 



X, y = spiral_data(samples=100, classes=3) 
dense1 = Layer_dense(2,3)
dense2 = Layer_dense(3,3)
dense1.forward_pass(X)

Activ = Activation_layer()
Activ_soft = Activation_softmax()
Activ.forward(dense1.output)

dense2.forward_pass(Activ.output)
Activ_soft.Softmax(dense2.output)
print(Activ_soft.output[:5])

Loss = Loss_crossEntropy()
loss = Loss.Calculate(Activ_soft.output, y)

print(loss)

