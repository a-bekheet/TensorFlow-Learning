import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 # data normalization

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # flattens to reduce dimensionality of 28x28 to 784
    keras.layers.Dense(128, activation='relu'), # 128 nodes in the hidden layer with relu activation
    keras.layers.Dense(10) # output layer with 10 nodes for 10 classes
    # do not include softmax here do it later in the loss function
])

# weight matrix is 784x128 and bias is 128 since each node in the dense layer has one bias
# number of nodes for a dense layer is (input_shape + 1) * number of nodes

# Loss and Optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # from_logits=True since we did not include softmax in the last layer
optimiser = keras.optimizers.Adam(learning_rate=0.001) # important hyperparameter
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optimiser, metrics=metrics)

# Training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)