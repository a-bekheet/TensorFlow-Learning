import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    return((x_train, y_train), (x_test, y_test))



def fashion_model():
    inputs = keras.Input(shape=(28, 28, 1))
    # layer 1
    x = layers.Conv2D(28, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    # layer 2
    x = layers.Conv2D(28*2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    # layer 3
    x = layers.Conv2D(28*4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    # layer 4
    x = layers.Conv2D(28*4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    # outputs
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

(x_train, y_train), (x_test, y_test) = load_data()
model = fashion_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(
        learning_rate=3e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
