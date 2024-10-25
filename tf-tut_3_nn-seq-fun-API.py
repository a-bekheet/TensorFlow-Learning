"""
Documentation: https://www.tensorflow.org/guide/keras/sequential_model
Sequential model API is the simplest way to build a model in Keras. It allows you to build a model layer by layer. Each layer has weights that correspond to the layer the follows it.
The Sequential model API is great for developing deep learning models in most situations, but it also has some limitations. For example, it does not allow you to create models that share layers or have multiple inputs or outputs.
The Functional API in Keras is a way to create models that are more flexible than the Sequential API. The Functional API can handle models with non-linear topology, models with shared layers, and models with multiple inputs or outputs.
The Functional API is defined by creating instances of layers and connecting them directly to each other in pairs, then defining a Model that specifies the layers to act as the input and output to the model.
The Functional API is more flexible than the Sequential API, but it is also more complex and requires more lines of code to define a model.
The Functional API is a great choice for building deep learning models that are more complex than a simple sequence of layers.
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
from tensorflow import keras  # high level API
from tensorflow.keras import layers  # type: ignore # Add this line
from tensorflow.keras.datasets import mnist  # type: ignore # dataset for handwritten digits

(x_train, y_train,), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # 10k images, 28x28 pixels
print(y_train.shape)

# We are going to send this data to a neural network so we want to flatten the 28x28 pixels
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0  # the -1 means keep whatever value is on that dimension
# so we are also changing it to float32 instead of the default float64 and normalizing to make the range between 0 and 1
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
# these will be numpy arrays the conversion to tf will happen internally

## Creating NN
# Sequential API (Very convenient, not very flexible) - Injective
model = keras.Sequential(
    [
        keras.Input(shape=(28*28,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10) # no activation since this is output layer
    ]
)

# Functional API (A bit more flexible)
inputs = keras.Input(shape=(784,)) # note the bracket around the shape
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile( #network configuration (loss function/optimizer)
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit( #concrete training of network
    x_train, y_train, batch_size=32, epochs=5, verbose=2
)

model.evaluate(
    x_test, y_test, batch_size=32, verbose=2
)

