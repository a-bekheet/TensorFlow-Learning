import numpy as np
import matplotlib.pyplot as plt

# Create a simple 4x4 input with clearer numbers
input_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Perform max pooling with 2x2 window
output_data = np.array([
    [6, 8],    # max of top-left 2x2     max of top-right 2x2
    [14, 16]   # max of bottom-left 2x2   max of bottom-right 2x2
])

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot input
ax1.imshow(input_data, cmap='viridis')
ax1.set_title('Input (4x4)')
for i in range(input_data.shape[0]):
    for j in range(input_data.shape[1]):
        ax1.text(j, i, str(input_data[i, j]), 
                ha='center', va='center', color='white')

# Add 2x2 grid to show pooling windows
ax1.grid(True, which='major', color='red', linewidth=2)
ax1.set_xticks([0.5, 2.5])
ax1.set_yticks([0.5, 2.5])

# Plot output
ax2.imshow(output_data, cmap='viridis')
ax2.set_title('After MaxPool2D (2x2)\nTakes maximum value from each 2x2 grid')
for i in range(output_data.shape[0]):
    for j in range(output_data.shape[1]):
        ax2.text(j, i, str(output_data[i, j]), 
                ha='center', va='center', color='white')

plt.tight_layout()
plt.show()