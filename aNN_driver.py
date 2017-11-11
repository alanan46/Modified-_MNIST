#!/usr/bin/env python
import numpy as np
# import scipy.misc # to visualize only
from sklearn.model_selection import train_test_split

from aNN import aNN
# import get_feat_fun as features

""" change the file to the actual ones"""
x=np.loadtxt("x1000.csv",delimiter=",")
y = np.loadtxt("y1000.csv", delimiter=",")
x = x.astype(np.float32, copy=False)
#normalize input_x
x=x/np.amax(x)
y = y.astype(np.int32, copy=False)
# train on 90% of the whole input data
train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.1, random_state=412)
#hidden layer=3, 50 neuron per hidden layer, learn_rate =0.1 epoch =100

testNN= aNN(train_data,train_labels,3,50,0.1,100)
"""note in the train function after it finishes, it will write a file and save it if u trained again without
changing the file path it will OVERWRITE
"""
testNN.train()
#get prediction
train_predict=[testNN.predict(x) for x in train_data]
eval_predict=[testNN.predict(x) for x in eval_data]
# # this was ~ 16 percent when I run based on the above  took about an hour
print("train accuracy:")
testNN.get_accuracy(train_predict,train_labels)
print("cross Validation accuracy:")
testNN.get_accuracy(eval_predict,eval_labels)
# print(testNN.weights)
# # w+ means overwrite the file if exist, can be changed
# with open("./weights.npy",'w+') as f:
#     np.save(f,testNN.weights)
