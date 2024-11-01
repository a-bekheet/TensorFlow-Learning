

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt

"""
Practice Problem: Fashion Item Classification with CNN

Task: Create a Convolutional Neural Network (CNN) to classify fashion items using 
the Fashion MNIST dataset, which is built into TensorFlow.

The goal is to build a model that can identify different types of clothing items:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

We'll implement:
1. Data preprocessing
2. CNN architecture with modern practices
3. Training with callbacks
4. Model evaluation and visualization
5. Predictions on new data
"""

# Load and preprocess data
def load_and_preprocess_data():
    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return (X_train, y_train), (X_test, y_test)

# Create CNN model
def create_model():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Training callbacks
def get_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

# Plotting functions
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = tf.math.confusion_matrix(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return plt.gcf()

"""
Practice Tasks:

1. Basic Implementation:
   - Load and preprocess the Fashion MNIST dataset
   - Create the CNN model using create_model()
   - Compile the model with appropriate loss function and optimizer
   - Train the model using the callbacks provided
   - Evaluate the model on the test set
   - Visualize the training history and confusion matrix

2. Model Improvement Tasks:
   - Experiment with different model architectures:
     * Add or remove convolutional layers
     * Try different numbers of filters
     * Modify the dense layer sizes
   - Test different optimizers (Adam, SGD with momentum, RMSprop)
   - Implement data augmentation using tf.keras.preprocessing.image.ImageDataGenerator
   - Add regularization techniques (L1/L2, increase/decrease dropout)

3. Advanced Tasks:
   - Implement transfer learning using a pre-trained model (e.g., MobileNetV2)
   - Create a custom training loop using tf.GradientTape
   - Implement gradient clipping
   - Add learning rate warmup
   - Create a custom callback for logging training metrics

Example implementation:
"""

def main():
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    return model, history

"""
Questions to Consider:

1. Architecture Design:
   - Why did we use two convolutional layers before each pooling layer?
   - What's the purpose of BatchNormalization after convolutions?
   - How does the dropout rate affect model performance?

2. Training Process:
   - How does the learning rate schedule affect training stability?
   - What's the impact of batch size on training time and final accuracy?
   - How effective is early stopping in preventing overfitting?

3. Model Evaluation:
   - Which classes does the model confuse most often and why?
   - How would you handle class imbalance if it existed in the dataset?
   - What metrics besides accuracy would be relevant for this problem?

Bonus Challenges:

1. Implement a custom layer that adds Gaussian noise during training
2. Create a feature visualization system to see what patterns activate each filter
3. Build a simple web interface using TensorFlow.js to deploy the model
4. Implement ensemble learning with multiple models
5. Create a custom metric that weighs certain types of misclassifications differently
"""