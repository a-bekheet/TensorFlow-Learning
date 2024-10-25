import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
from tensorflow import keras  # high level API
from tensorflow.keras import layers  # type: ignore # Add this line
from tensorflow.keras.datasets import mnist  # type: ignore # dataset

(x_train, y_train,), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# images of 28x28 pixels for each timestep we will unroll the image in rows of 28 pixels
# we should not use RNNs for images this is for illustrative purposes

model = keras.Sequential() # we do not need to specify a specific number of timesteps
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation='tanh')
        ) # its returning output from each timestep for stacking
)
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation='tanh')
        ) # doubles the nodes one forwards one backwards
)
model.add(layers.Dense(10))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=1)