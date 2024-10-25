import time
import numpy as np
import tensorflow as tf
import platform
import psutil
import multiprocessing

def get_system_info():
    """Get system information including GPU details"""
    print("System Information:")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024.0 ** 3):.1f} GB")
    print(f"Python Version: {platform.python_version()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("\nGPU Information:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"GPU: {gpu.device_type} - {gpu.name}")
    else:
        print("No GPU detected")
    print("")

def cpu_matrix_multiply(size=2000):
    """Matrix multiplication on CPU using NumPy"""
    print(f"\nRunning CPU matrix multiplication ({size}x{size})...")
    start = time.time()
    
    # Create random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Multiply matrices
    C = np.dot(A, B)
    
    duration = time.time() - start
    print(f"CPU Time: {duration:.2f} seconds")
    return duration

def gpu_matrix_multiply(size=2000):
    """Matrix multiplication on GPU using TensorFlow"""
    print(f"\nRunning GPU matrix multiplication ({size}x{size})...")
    start = time.time()
    
    # Create random matrices on GPU
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        A = tf.random.uniform((size, size))
        B = tf.random.uniform((size, size))
        
        # First run to warm up GPU
        C = tf.matmul(A, B)
        tf.keras.backend.clear_session()
        
        # Actual timed run
        start = time.time()
        C = tf.matmul(A, B)
        # Force execution by accessing result
        C.numpy()
        
    duration = time.time() - start
    print(f"GPU Time: {duration:.2f} seconds")
    return duration

def convolutional_operation_benchmark(size=1000, channels=32):
    """Test convolution operations (common in deep learning)"""
    print(f"\nRunning convolution operations benchmark...")
    
    # Create input with shape (batch_size, height, width, channels)
    input_shape = (1, size, size, channels)
    
    # CPU benchmark
    print("CPU Convolution:")
    with tf.device('/CPU:0'):
        x_cpu = tf.random.uniform(input_shape)
        conv_cpu = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')
        
        # Warmup
        _ = conv_cpu(x_cpu)
        
        # Timed run
        start = time.time()
        output_cpu = conv_cpu(x_cpu)
        output_cpu.numpy()  # Force execution
        cpu_time = time.time() - start
        print(f"CPU Time: {cpu_time:.2f} seconds")
    
    # GPU benchmark
    if tf.config.list_physical_devices('GPU'):
        print("GPU Convolution:")
        with tf.device('/GPU:0'):
            x_gpu = tf.random.uniform(input_shape)
            conv_gpu = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')
            
            # Warmup
            _ = conv_gpu(x_gpu)
            
            # Timed run
            start = time.time()
            output_gpu = conv_gpu(x_gpu)
            output_gpu.numpy()  # Force execution
            gpu_time = time.time() - start
            print(f"GPU Time: {gpu_time:.2f} seconds")
    else:
        gpu_time = float('inf')
    
    return cpu_time, gpu_time

def run_benchmarks():
    """Run all benchmarks and display results"""
    get_system_info()
    
    # Matrix multiplication benchmarks
    sizes = [1000, 2000, 4000]
    for size in sizes:
        cpu_time = cpu_matrix_multiply(size)
        gpu_time = gpu_matrix_multiply(size)
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"GPU is {speedup:.1f}x faster than CPU")
        else:
            slowdown = gpu_time / cpu_time
            print(f"GPU is {slowdown:.1f}x slower than CPU")
    
    # Convolution benchmark
    cpu_conv_time, gpu_conv_time = convolutional_operation_benchmark()
    
    if gpu_conv_time < float('inf'):
        speedup = cpu_conv_time / gpu_conv_time
        print(f"\nConvolution Operations GPU Speedup: {speedup:.1f}x")
    
    print("\nBenchmark Summary:")
    print("Matrix Multiplication:")
    for i, size in enumerate(sizes):
        print(f"- {size}x{size} matrix size")
    print("\nConvolution Operations:")
    print(f"- 1000x1000 with 32 channels -> 64 filters")

if __name__ == "__main__":
    run_benchmarks()