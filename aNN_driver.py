#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split

from aNN import aNN
from get_feat_fun import get_features as features

print "Loading data..."
x = np.loadtxt("train_x.csv",delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",")
x = x.astype(np.float32, copy=False)
y = y.reshape(-1)
x = x.reshape(-1,64,64)
y = y.astype(np.int32, copy=False)
print "Getting features..."
X = features(x,'zernike')
# train on 90% of the whole input data
train_data, eval_data, train_labels, eval_labels = train_test_split(X, y, test_size=0.2,
                                                                    random_state=41, stratify = y)

print "Training model..."
#hidden layer=3, 50 neuron per hidden layer, learn_rate =0.1 epoch =100
testNN= aNN(train_data,train_labels,3,90,0.001,300)
"""note in the train function after it finishes, it will write a file and save it if u trained again without
changing the file path it will OVERWRITE
"""
testNN.train()
#get prediction
train_predict=[testNN.predict(x) for x in train_data]
eval_predict=[testNN.predict(x) for x in eval_data]
print("train accuracy:")
testNN.get_accuracy(train_predict,train_labels)
print("cross Validation accuracy:")
testNN.get_accuracy(eval_predict,eval_labels)
# print(testNN.weights)
# # w+ means overwrite the file if exist, can be changed
# with open("./weights.npy",'w+') as f:
#     np.save(f,testNN.weights)
