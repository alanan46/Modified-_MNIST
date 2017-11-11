#!/usr/bin/env python
import numpy as np
# import scipy.misc # to visualize only
from sklearn.model_selection import train_test_split

from toyaNN import aNN
# import get_feat_fun as features

# x=np.load("./zernike_features.npy")

# y = np.loadtxt("train_y.csv", delimiter=",")
# x=np.array([1,2,3])
# x = x.astype(np.float32, copy=False)
# y = y.astype(np.int32, copy=False)
# train on 90% of the whole input data
# train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.1, random_state=412)
#hidden layer=3, 100 neuron per hidden layer, learn_rate =0.1 epoch =3000
train_data=np.array([[1,0,0],[0,1,0],[0,0,1]])
train_labels=np.array([1,2,3])
testNN= aNN(train_data,train_labels,2,3,0.1,2000)
testNN.train()
train_predict=[testNN.predict(x) for x in train_data]
# eval_predict=[testNN.predict(x) for x in eval_data]
# # this was ~ 16 percent when I run based on the above  took about an hour
print("train accuracy:")
testNN.get_accuracy(train_predict,train_labels)
# print("cross Validation accuracy:")
# testNN.get_accuracy(eval_predict,eval_labels)
# print(testNN.weights)
# # w+ means overwrite the file if exist, can be changed
# with open("./weights.npy",'w+') as f:
#     np.save(f,testNN.weights)
