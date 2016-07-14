from __future__ import print_function
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random

# Loading SVHN Train and Test Data
X_Train = loadmat('train_32x32.mat')['X'].astype('float64')
y_Train = loadmat("train_32x32.mat")['y'].astype('int8')
y_Testing = loadmat("test_32x32.mat")['y'].astype('int8')

fig = plt.figure()
for i in np.arange(20):
    j = random.choice(np.arange(len(X_Train)))
    plot = fig.add_subplot(4, 5, i+1)
    plt.title(y_Train[j])
    plt.imshow(X_Train[: , :, :, j])
    plt.axis('off')
plt.show()

# Replace class 10 with 0 as it originally represents 0
# This will enable us to tranform it into one-hot encoding
np.place(y_Train, y_Train == 10, 0)
np.place(y_Testing, y_Testing == 10, 0)

unique, counts = np.unique(y_Train, return_counts=True)

plt.bar(unique, counts)
pos = np.arange(len(unique))
plt.xticks(pos + 0.4, unique)
plt.show()

x = dict(zip(unique, counts))

print(x)

unique, counts = np.unique(y_Testing, return_counts=True)

plt.bar(unique, counts)
pos = np.arange(len(unique))
plt.xticks(pos + 0.4, unique)
plt.show()

y = dict(zip(unique, counts))

print(y)

