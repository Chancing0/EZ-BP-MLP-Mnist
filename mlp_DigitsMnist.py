# -*- coding: utf-8 -*-
"""
@author: Chancing
"""
import random
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn import datasets

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix 
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

#  sigmoid function,  standard 1/(1+e^-x)
def sigmoid(x):
    
    return (1/(1+np.exp(-x)))

# derivative of our sigmoid function
def dsigmoid(x):
    
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni
        self.nh = nh
        self.no = no
        self.nhb = nh #number of bias of hidden layer
        self.nob = no #number of bias of output layer

        # activations for nodes
        # create an initial list for each layer
        # you can use "print('self.ai=', self.ai)" to show the list
        self.ai = [1.0]*self.ni #[3.0]*3=[3.0, 3.0, 3.0]
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        self.bh = [1.0]*self.nhb #bias list of hidden layer
        self.bo = [1.0]*self.nob #bias list of output layer
        
#        self.zh = [1.0]*self.nh
#        self.zo = [1.0]*self.no       
       
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)      
        
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2) #range of weight from i to j layer stores in matrix wi
        for j in range(self.nh):                
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        for l in range(self.nh):
            self.bh[l] = rand(-0.3, 0.3)
        for m in range(self.no):
            self.bo[m] = rand(-0.3, 0.3)


    def feedforward(self, inputs):
        
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        # for the input layer, just output the value to the next layer
        for i in range(self.ni): 
            self.ai[i] = inputs[i] 

        # hidden activations
        for j in range(self.nh):
            # summation the (input * weight), for all training examples
            # then plus bias
            sum = 0.0
            for i in range(self.ni):
                sum+=self.ai[i]*self.wi[i][j]
                #a=1
            #Write down your code below
#            self.zh[j]=sum+self.bh[j]
            
            self.ah[j]=sigmoid(sum+self.bh[j])

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum+=self.ah[j]*self.wo[j][k]
                #a=1
            #Write down your code below
            
#            self.zo[k]=sum+self.bh[k]
            
            self.ao[k]=sigmoid(sum+self.bo[k])
        
        
        return self.ao 


    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no # [0.0]*3=[0.0, 0.0, 0.0] for initial output_deltas
        for k in range(self.no):
            error = -(targets[k]-self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error
#            output_deltas[k] = error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                #Write down your code below
                error += output_deltas[k] * self.wo[j][k]
            #Write down your code below
            
            hidden_deltas[j] = dsigmoid(self.ah[j])* error 
            
            
        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                #Write down your code below
                self.wo[j][k]= self.wo[j][k] - N*output_deltas[k]*self.ah[j]

              
        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                #Write down your code below
                self.wi[i][j]= self.wi[i][j] - N*hidden_deltas[j]*self.ai[i]
 
        
        # update output bias
        for j in range(self.no):
            self.bo[j] =self.bo[j] - N*output_deltas[j]
            
        # update hidden bias
        for j in range(self.nh):
            self.bh[j] =self.bh[j] - N*hidden_deltas[j]
        
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, data):
        for p in data:
            print(p[0], '->', self.feedforward(p[0]))


    def train(self, data, iterations=1000, alpha=0.01):
        # N: learning rate
        # M: momentum factor
        # iteration: no. of iterations
        for i in range(iterations):
            error = 0.0
            
            '''
            patterns is the training data
            patterns = [[[0, 0] [0]]
                        [[0, 1] [1]]
                        [[1, 0] [1]]
                        [[1, 1] [0]]]
            the line 'for p in patterns:'
            p is the raw of patterns, p[0] is x, p[1] is y 
            '''
            for p in data:
                x = p[0]
                y = p[1]
                self.feedforward(x)
                error = error + self.backPropagate(y, alpha)
            
            if i % 100 == 0:
                pylab.plot(i,error,'o')
                print('error %-.5f' % error)


def demo():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    skldata = digits.images.reshape((n_samples, -1))
    T = np.zeros((digits.target.shape[0],10))
    T[np.arange(len(T)), digits.target] += 1

    whole_data=[]
    for a,b, in zip(skldata, T):
        whole_data = whole_data + [[a,b]]
    training_no = 100
    test_no = 589
    d = whole_data[0:training_no]
    d_test = whole_data[test_no:test_no+1] 
    plt.subplot(1,2,1)
    pylab.imshow(digits.images[test_no], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.subplot(1,2,2)
    print('d_test=',d_test)
    # create a network with two input, two hidden, and one output nodes
    n = NN(64, 200, 10)
    # train it with some data
    n.train(d)
    # test the same data
    n.test(d_test)
    
    pylab.show()
    

if __name__ == '__main__':
    demo()
