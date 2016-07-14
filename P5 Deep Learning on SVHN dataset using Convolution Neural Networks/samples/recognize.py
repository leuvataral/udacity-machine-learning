from __future__ import print_function
from keras.models import model_from_json
from scipy.misc import imread
import numpy as np
import sys

print("Processing...")

model = model_from_json(open('keras_svhn_architecture.json').read())
model.load_weights('keras_svhn_weights.h5')

model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

filename = str(sys.argv[1])

X = imread(filename)

X = X.transpose()

X = X.reshape(1, 3, 32, 32)

pred = model.predict(X)

print("Recognized number:", np.argmax(pred[0]))
