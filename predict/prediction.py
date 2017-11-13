import keras
import numpy as np
import pandas as pd
from keras.models import load_model
model = load_model('final2_model.h5')
x = np.loadtxt("test_x.csv", delimiter=",") # load from text
x = x.reshape(-1,1, 64, 64) # reshape
x=x.astype('float32')
#normalization to 0-1
x=x/255.0
#this is classes fitted
y_class=model.predict_classes(x)
label_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25,
 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
#onehot decoding
y=[label_class[ele] for ele in y_class ]

""" output the csv file"""
def CSVify(myList,name):
    df=pd.DataFrame(myList)
    df=df.reset_index()
    df['index']=df['index']+1
    df.to_csv(name+'.csv',header=['Id','LABEL'],index=False)

CSVify(y,"final2")
