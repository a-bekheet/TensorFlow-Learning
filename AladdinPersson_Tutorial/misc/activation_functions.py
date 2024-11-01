import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-5, 5, 100)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Create plot
plt.figure(figsize=(12, 8))

plt.plot(x, relu(x), label='ReLU', linewidth=2)
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.plot(x, tanh(x), label='Tanh', linewidth=2)
plt.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2)

plt.grid(True)
plt.legend(fontsize=12)
plt.title('Common Activation Functions', fontsize=14)
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.show()