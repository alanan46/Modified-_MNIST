# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from get_feat_fun import get_features as features
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
print "Loading data..."
x = np.loadtxt("train_x.csv", delimiter=",") # load from text
#x_test = np.loadtxt("test_x.csv", delimiter=",") # load from text
y = np.loadtxt("train_y.csv", delimiter=",")
y = y.reshape(-1)
x = x.reshape(-1,64,64)
#x_test = x_test.reshape(-1,64,64)
#x = x/np.max(x)

print "Getting features..."
XX = features(x,'zernike')
#XX_test = get_features(x_test,'zernike')
Xtrain, Xval, Ytrain, Yval = train_test_split(XX, y,
                                                    test_size=0.2,random_state=45)

# Single run
nn = MLPClassifier(hidden_layer_sizes=(90, 3), activation='relu',
                   solver='sgd', learning_rate='constant',  random_state=43,
                   max_iter=300)
print "Training model..."
nn.fit(Xtrain,Ytrain)
acc = nn.score(Xval, Yval)
#pred = nn.predict(Xval)
print("Validation accuracy: {:.2f}%".format(acc * 100))

# Cross-Validation run for hyperparameter tuning
#VA = []
#HL = []
#N = []
#Lr = []
#for hl in range(1,6):
#    for n in range(10,110,10):
#        for lr in np.arange(0.001,0.5,0.001):
#            nn = MLPClassifier(hidden_layer_sizes=(35, 4), activation='relu',
#                               alpha=1e-5, solver='sgd', learning_rate='constant',
#                               max_iter=300)
#            print "Training model..."
#            nn.fit(Xtrain,Ytrain)
#            acc = nn.score(Xval, Yval)
#            pred = nn.predict(Xval)
#            print("Validation accuracy: {:.2f}%".format(acc * 100))
#            HL.append(hl)
#            N.append(n)
#            Lr.append(lr)
#            VA.append(acc)