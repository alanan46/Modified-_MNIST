import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
K.set_image_dim_ordering('th')
np.random.seed(10)
x = np.loadtxt("half2_train_x.csv", delimiter=",") # load from text
y = np.loadtxt("half2_train_y.csv", delimiter=",")
x = x.reshape(-1,1, 64, 64) # reshape
x=x.astype('float32')
#normalization to 0-1
x=x/255.0
y = y.reshape(-1)
#one hot encoding
encoder=LabelEncoder()
y=encoder.fit(y)
y=np_utils.to_categorical(y)
train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.1, random_state=42)
num_classes = 40

# Create the model
model = Sequential()
model.add(Conv2D(20, (3, 3), input_shape=(1, 64, 64), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(30, (3, 3),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
model.add(Conv2D(30, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(60, (3, 3),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(300, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores1 = model.evaluate(train_data, train_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))

scores2 = model.evaluate(eval_data, eval_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
