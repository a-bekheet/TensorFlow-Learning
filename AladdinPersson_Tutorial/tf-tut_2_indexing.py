import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])
x = tf.ones((3,3))
x = tf.zeros((2,3))
x = tf.eye(3) # I for the identity matrix (eye)

## for distributions

x = tf.random.normal((3,3), mean = 0, stddev = 1)

x = tf.random.uniform((1,3), minval = 0, maxval = 3)
x = tf.range(start=1, limit=10, delta=2)

x = tf.cast(x, dtype=tf.float64) # convert types
# tf.float (16,32,64), tf.int (8..64), tf.bool

# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([7,8,9])

z = tf.add(x, y)
z = tf.subtract(x, y)
z = tf.divide(x, y)
z = tf.multiply(x, y)

z = tf.tensordot(x, y, axes=1)

# Indexing
x = tf.constant([0, 1, 2, 3, 4, 5])
# print(x[:]) # all
# print(x[1:]) # from 1 to end
# print(x[1:3]) # from 1 to 3 noninclusive
# print(x[::2]) # every 2nd element
# print(x[::-1]) # reverse

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)

x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])

print(x[0:2, :])
