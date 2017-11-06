#!/usr/bin/env python
import numpy as np
# import scipy.misc # to visualize only
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import KernelPCA, PCA
from NN import NN
# import get_feat_fun as features
a=[1,0,0]
b=[0,1,0]
c=[0,0,1]
x=[]
y=[]
for i in xrange(30):
    if i<10:
        x.append(a)
        y.append(a)
    elif i<20:
        x.append(b)
        y.append(b)
    else:
        x.append(c)
        y.append(c)
y=np.array(y)
# y = y.reshape(-1,1)
x=np.array(x)
x = x.astype(np.float32, copy=False)
y = y.astype(np.int32, copy=False)
# train on 90% of the whole input data
train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.2, random_state=412)
#hidden layer=1, 10 neuron per hidden layer, learn_rate =0.5 epoch =4
testNN= NN(train_data,train_labels,1,3,0.1,3000)
testNN.train()
testNN.predict(eval_data[2])
print(eval_labels[2])
