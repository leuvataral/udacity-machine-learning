from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils
from keras.models import model_from_json

nb_classes = 10

np.random.seed(1010)

# Loading SVHN Train and Test Data
X_Test = loadmat("test_32x32.mat")['X'].astype('float64')
y_Test = loadmat("test_32x32.mat")['y'].astype('int8')

# Transpose the dataset for changing the shape to (.., 3, 32, 32)
X_Test = X_Test.transpose()

# Scaling pixel values by dividing each by 255
X_Test /= 255

# Replace class 10 with 0 as it originally represents 0
# This will enable us to tranform it into one-hot encoding
np.place(y_Test, y_Test == 10, 0)

# Transform class labels into one-hot encoding
y_TestB = np_utils.to_categorical(y_Test, nb_classes)

# Load the earlier trained model and the weights
model = model_from_json(open('keras_svhn_architecture.json').read())
model.load_weights('keras_svhn_weights.h5')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

# Test the model on the whole test dataset
score = model.evaluate(X_Test, y_TestB, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
