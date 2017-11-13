import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from get_feat_fun import get_features as features
print "Loading data..."
x = np.loadtxt("train_x.csv", delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",")
x = x.reshape(-1,64,64)
y = y.reshape(-1,1)

print "Getting features..."
XX = features(x,'zernike')

Xtrain, Xval, Ytrain, Yval = train_test_split(XX, y,
                                              test_size=0.25,random_state=44)

print "Training model..."
# Logistic Regression Model
logreg = linear_model.LogisticRegression()
logreg.fit(Xtrain, Ytrain)
acc = logreg.score(Xval, Yval)
print("Validation accuracy: {:.2f}%".format(acc * 100))


# Applying the model of choice on the test data
#Pred_labels = logreg.predict(zernike_features_norm_test)
#Pred = Pred_labels.astype(int)
#with open("Output_4.csv", 'wb') as f:
#    writer = csv.writer(f, delimiter=",")
#    writer.writerow(list(Pred))
