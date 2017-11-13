import numpy as np
import keras
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
from tensorflow.python.lib.io import file_io
from datetime import datetime
import time
import argparse

K.set_image_dim_ordering('th')
np.random.seed(10)
def train(x_file="train_x.csv",y_file="train_y.csv",job_dir="./tmp/keras",**args):
    log_path=job_dir+'/logs/'+datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(x_file))
    print('Using logs_path located at {}'.format(log_path))
    print('-----------------------')
    file_stream_x = file_io.FileIO(x_file, mode='r')
    file_stream_y = file_io.FileIO(y_file, mode='r')
    x = np.loadtxt(file_stream_x, delimiter=",") # load from text
    y = np.loadtxt(file_stream_y, delimiter=",")

    x = x.reshape(-1,1, 64, 64) # reshape
    x=x.astype('float32')
    #normalization to 0-1
    x=x/np.amax(x)
    y = y.reshape(-1)
    #one hot encoding
    encoder=LabelEncoder()
    encoder.fit(y)
    encoded_y=encoder.transform(y)
    y=np_utils.to_categorical(encoded_y)
    train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.2, random_state=42)
    num_classes = 40

    # Create the model
    model = Sequential()

    model.add(Conv2D(16, (5, 5), input_shape=(1, 64, 64), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (5, 5),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Conv2D(64, (3, 3),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 250
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=job_dir+"/graph", histogram_freq=0, write_graph=True, write_images=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    # Fit the model
    model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels), epochs=epochs, batch_size=128,callbacks=[tbCallBack])
    # Final evaluation of the model
    scores1 = model.evaluate(train_data, train_labels, verbose=0)
    print("Test Loss: %.2f%%" % (scores1[0]*100))
    print("Test Accuracy: %.2f%%" % (scores1[1]*100))
    model.save("model.h5")
        # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--x_file',
      help='GCS or local paths to training data x',
      required=True
    )
    parser.add_argument(
      '--y_file',
      help='GCS or local paths to training data y',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train(**arguments)
