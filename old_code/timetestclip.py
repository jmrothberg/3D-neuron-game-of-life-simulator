import numpy as np
import time

def sigmoid_no_clip(x):
    return 1 / (1 + np.exp(-x))

def sigmoid(x): #use for forward
    # Prevent overflow.
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

def test_function():
    x = np.random.randn(1000000)  # 1 million random numbers

    start_time_with_clip = time.time()
    _ = sigmoid(x)
    end_time_with_clip = time.time()
    print(f"Time with clip: {end_time_with_clip - start_time_with_clip} seconds")

    start_time_no_clip = time.time()
    _ = sigmoid_no_clip(x)
    end_time_no_clip = time.time()
    print(f"Time without clip: {end_time_no_clip - start_time_no_clip} seconds")

test_function()