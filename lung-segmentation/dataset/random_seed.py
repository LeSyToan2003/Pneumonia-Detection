import numpy as np
import tensorflow as tf
import keras
import random

def random_seed(seed):
    
    """
    Sets a fixed random seed for reproducibility of results.

    This function ensures that the random operations from different libraries (e.g., Python's built-in random,
    NumPy, TensorFlow, and Keras) generate the same results every time the code is run. This is useful for debugging
    or comparing experiments in machine learning to get consistent results.

    Parameters:
    - seed: The fixed integer seed value to set the random state across different libraries.
    """
    
    
    # Set the random seed for Python's built-in random module
    random.seed(seed)
    
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed for TensorFlow to ensure the reproducibility of operations in TensorFlow
    tf.random.set_seed(seed)
    
    # Set the random seed for Keras (this will affect Keras operations that rely on randomness)
    keras.utils.set_random_seed(seed)