import time
import numpy as np
import platform
import psutil
import multiprocessing
from math import sqrt

def cpu_float_operations(n=1000000):
    """Test floating point operations"""
    start = time.time()
    for i in range(n):
        sqrt(i * 2.5) / 2.7
    return time.time() - start

def numpy_matrix_operations(size=1000):
    """Test matrix operations using NumPy"""
    start = time.time()
    matrix1 = np.random.rand(size, size)
    matrix2 = np.random.rand(size, size)
    result = np.dot(matrix1, matrix2)
    return time.time() - start

def prime_finder(n=10000):
    """Test integer operations by finding prime numbers"""
    start = time.time()
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return time.time() - start

def run_benchmark():
    print("System Information:")
    print(f"CPU: {platform.processor()}")
    print(f"Cores: {multiprocessing.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024.0 ** 3):.1f} GB")
    print(f"Python Version: {platform.python_version()}")
    print("\nRunning benchmarks...\n")

    # Run tests
    float_time = cpu_float_operations()
    matrix_time = numpy_matrix_operations()
    prime_time = prime_finder()

    print("Benchmark Results:")
    print(f"Floating Point Operations (1M): {float_time:.2f} seconds")
    print(f"Matrix Multiplication (1000x1000): {matrix_time:.2f} seconds")
    print(f"Prime Number Finding (up to 10K): {prime_time:.2f} seconds")

    # Provide rough performance assessment
    total_time = float_time + matrix_time + prime_time
    
    print("\nPerformance Assessment:")
    if total_time < 1.5:
        print("Excellent performance! Your computer is very fast.")
        print("Typical for: High-end desktop/laptop with modern CPU")
    elif total_time < 3:
        print("Good performance. Your computer is above average.")
        print("Typical for: Mid-range modern desktop/laptop")
    elif total_time < 5:
        print("Average performance.")
        print("Typical for: Standard laptop or older desktop")
    else:
        print("Below average performance.")
        print("Typical for: Older or budget computers")

    print("\nNote: These benchmarks are simplified and results may vary based on background processes")

if __name__ == "__main__":
    run_benchmark()