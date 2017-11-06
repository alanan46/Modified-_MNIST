#!/usr/bin/env python
import numpy as np
import scipy.misc # to visualize only
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA, PCA
from NN import NN
import get_feat_fun as features

x = np.loadtxt("x1000.csv", delimiter=",") # load from text
y = np.loadtxt("y1000.csv", delimiter=",")
y = y.reshape(-1,1)
x = x.reshape(-1, 64, 64) # reshape
x=features.get_features(x,"zernike",True)
x=np.array(x)
x = x.astype(np.float32, copy=False)
y = y.astype(np.int32, copy=False)
# train on 90% of the whole input data
train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.1, random_state=42)
#hidden layer=1, 10 neuron per hidden layer, learn_rate =0.5 epoch =4
testNN= NN(train_data,train_labels,1,10,0.5,4)
testNN.train()
