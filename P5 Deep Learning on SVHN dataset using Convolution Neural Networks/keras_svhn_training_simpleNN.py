from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D
from sklearn.cross_validation import train_test_split

np.random.seed(1010)
batch_size = 128
nb_classes = 10
nb_epoch = 20

# Image dimensions
img_rows, img_cols = 32, 32
# Number of convolution filters to use
nb_filters = 32
# Convolution kernel size
nb_conv = 3

# Loading SVHN Train and Test Data
X_Train = loadmat('train_32x32.mat')['X'].astype('float64')
y_Train = loadmat("train_32x32.mat")['y'].astype('int8')
X_Testing = loadmat("test_32x32.mat")['X'].astype('float64')
y_Testing = loadmat("test_32x32.mat")['y'].astype('int8')



# Transpose the dataset for changing the shape to (.., 3, 32, 32)
X_Train = X_Train.transpose()
X_Testing = X_Testing.transpose()

# Split test_32x32.mat into test and validation data
X_Test, X_Valid, y_Test, y_Valid = train_test_split(X_Testing, y_Testing, test_size=0.5)

# Scaling pixel values by dividing each by 255
X_Train /= 255
X_Test /= 255
X_Valid /= 255

# Replace class 10 with 0 as it originally represents 0
# This will enable us to tranform it into one-hot encoding
np.place(y_Train, y_Train == 10, 0)
np.place(y_Test, y_Test == 10, 0)
np.place(y_Valid, y_Valid == 10, 0)

# Transform class labels into one-hot encoding
y_TrainB = np_utils.to_categorical(y_Train, nb_classes)
y_TestB = np_utils.to_categorical(y_Test, nb_classes)
y_ValidB = np_utils.to_categorical(y_Valid, nb_classes)

# Defining the model
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

# Train the model
model.fit(X_Train, y_TrainB, batch_size=batch_size, nb_epoch=nb_epoch,
          shuffle=True, verbose=1, validation_data=(X_Valid, y_ValidB))

# Evaluate the trained model on the whole test data
score = model.evaluate(X_Test, y_TestB, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
