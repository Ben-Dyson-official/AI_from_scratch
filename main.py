import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

data = np.array(pd.read_csv('/Users/bendyson/Coding/gitRepos/AI_from_scratch/mnist_train.csv'))

m, n = data.shape

np.random.shuffle(data)
test_data = data[0:1000].T


Y_test = test_data[0]
X_test = test_data[1:n]


train_data = data[1000:m].T
Y_train = train_data[0]
X_train = train_data[1:n]

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 784)
    b2 = np.random.rand(10, 1)

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    np.exp(Z) / np.sum(np.exp(Z))
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(X) + b2
    A2 = softmax(Z2)









