#!/usr/bin/env python
# coding:utf-8
"""
@version: 
@author: xiaojianli
@contact:xiaoli_lixj@163.com
@software: PyCharm
@time:  上午10:04
"""
import tensorflow as tf
import numpy as np

#create variables
x_data = np.random.rand(100)  #create dataset
y_data = x_data * 0.1 + 0.3 #predict; set the rule ; 0.1 is the pre weight, 0.3 is the biase

## create the constructor of tensor graph

# weights = tf.Variable(np.random.rand([1], -0.1, 1))
weights = tf.Variable(tf.random_uniform([1], -0.1, 1, dtype=np.float32)) #create the weight tensor ;dimention is 1, the weights should be -0.1 to 1
biases = tf.Variable(tf.zeros([1]), dtype=np.float32) #create the biases tensor

y = x_data * weights + biases #operate the rule

#define loss
loss = tf.reduce_mean(tf.square(y_data - y))    # (y_data - y)^2 / len(x_data)
optimizer = tf.train.GradientDescentOptimizer(0.5) #use the GDO optimizer-> then make the loss be minimum; the learning rate is 0.5
train = optimizer.minimize(loss)    #the optimizer make the loss be minimum
init = tf.initialize_all_variables()    #initialize all varibles

##tensor constructor end

sess = tf.Session() #define a session
sess.run(init)

for step in range(200):
    sess.run(train)
    if(step % 20 == 0):
        print(step , sess.run(weights) , sess.run(biases), sess.run(loss))


