# BP_Mnist_ezCode_MLP
ezCode of Backpropagation with sigle hidden layer to train Mnist and some Gate.  
Please pip install sklearn matplotlib pylab numpy 
# Run mlp_DigitsMnist.py  
just run , but you could change the model hyperparameters by below code line  
       def train(self, data, iterations=1000, alpha=0.01)  
alpha: learning rate  
iterations:The training times of the model you want.  
# Run mlp_Gate.py  
1. just change the data inside below  
def demo():  
      data = np.array([  
        [[0,0,0], [0]],  
        [[0,1,0], [1]],  
        [[1,0,0], [1]],  
        [[1,1,1], [0]]  
    ])    
as you can see , here is four samples, each samlpe has 3 inputs feature , and one lable.  
it's XOR gate Truth table with extended feture(X1*X2)  
2. change the hyperparameters baseon your data form of below code to built your NN  
    n = NN(3, 10, 1)  
it means input layer with 3 node of feature . single hidden layer with 10 nodes, output layer with 1 output.  
3. just run , but you could change the model hyperparameters by below code line  
       def train(self, data, iterations=1000, alpha=0.01):  
alpha: learning rate  
iterations:The training times of the model you want.  
