from __future__ import print_function
from keras.models import model_from_json
from keras.utils.visualize_util import plot

# Load the earlier trained model and the weights
model = model_from_json(open('keras_svhn_architecture.json').read())
model.load_weights('keras_svhn_weights.h5')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

# Plot model graph and save it to file model.png
plot(model, to_file='model.png')