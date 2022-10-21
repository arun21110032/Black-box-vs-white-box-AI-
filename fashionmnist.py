# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:44:43 2022

@author: Arun karthik
"""

import numpy as np 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train[:5000] / 255.0, x_test[:1000] / 255.0

x = np.array(x_train)
x_t = np.array(x_test)

y = np.array(y_train[:5000])
y_t = np.array(y_test[:1000])


# activation function

def sigmoid(x):
	return(1/(1 + np.exp(-x)))

# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)

def f_forward(x, w1, w2):
	# hidden
	z1 = x.flatten().dot(w1)# input from layer 1
	a1 = sigmoid(z1)# out put of layer 2
	
	# Output layer
	z2 = a1.dot(w2)# input of out layer
	a2 = sigmoid(z2)# output of out layer
	return(a2)

# initializing the weights randomly
def generate_wt(x, y):
	l =[]
	for i in range(x * y):
		l.append(np.random.randn())
	return(np.array(l).reshape(x, y))
	
# for loss we will be using mean square error(MSE)
def loss(out, y):
	s =(np.square(out-y))
	s = np.sum(s)/5000
	return(s)

# Back propagation of error
def back_prop(x, y, w1, w2, alpha,i):
    if(i%1000==0):
        print(i)
    x=x.flatten()
    z1 = x.dot(w1)
    a1 = sigmoid(z1)# output of layer 2
     
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    # error in output layer
    d2 =(a2-y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),(np.multiply(a1, 1-a1)))
    
    x_ = x.reshape(-1,x.shape[0])
    #print(x_.transpose().shape)
    
    d1_ = d1.reshape(-1,d1.shape[0])
    #print(d1_.shape)
    
    a1_=a1.reshape(a1.shape[0],-1)
    
    d2_ = d2.reshape(-1,d2.shape[0])
    #print(d2_.shape)
    
    # Gradient for w1 and w2
    w1_adj = x_.transpose().dot(d1_)
    w2_adj = a1_.dot(d2_)
     
    # Updating parameters
    w1 = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))
     
    return(w1, w2)

def train(x, Y, w1, w2, alpha = 0.01, epoch = 10):
	acc =[]
	losss =[]
	for j in range(epoch):
		l =[]
		for i in range(len(x)):
			out = f_forward(x[i].flatten(), w1, w2)
			l.append((loss(out, Y[i])))
			w1, w2 = back_prop(x[i], y[i], w1, w2, alpha,i)
		print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)
		acc.append((1-(sum(l)/len(x)))*100)
		losss.append(sum(l)/len(x))
	return(acc, losss, w1, w2)

def predict(x, w1, w2):
    out = f_forward(x, w1, w2)
    z=np.argmax(out)
    print(z)

w1 = generate_wt(784, 1000)
w2 = generate_wt(1000, 10)
print(w1, "\n\n", w2)

acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)
"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
predict(x[1], w1, w2)