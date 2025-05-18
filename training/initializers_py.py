"""
Weight initialization strategies for neural networks.

Proper weight initialization is critical for training deep neural networks.
This module provides implementations of modern initialization techniques
that help with faster convergence and avoiding vanishing/exploding gradients.
"""

import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):
    """
    Abstract base class for weight initializers.
    
    Weight initializers provide strategies for setting the initial values of
    the trainable parameters in a neural network.
    """
    @abstractmethod
    def initialize(self, shape):
        """
        Initialize weights with a specific strategy.
        
        Args:
            shape: The shape of the weights to initialize
            
        Returns:
            Initialized weights of the specified shape
        """
        pass

class HeInitializer(Initializer):
    """
    He (or Kaiming) initialization for weights.
    
    This initialization is particularly well-suited for layers with ReLU
    activations. It draws samples from a truncated normal distribution
    with mean 0 and standard deviation sqrt(2/fan_in).
    
    Reference: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level
    Performance on ImageNet Classification", 2015.
    """
    def initialize(self, shape):
        """
        Initialize weights using He initialization.
        
        Args:
            shape: The shape of the weights to initialize
            
        Returns:
            Initialized weights of the specified shape
        """
        # fan_in is the number of input units in the weight tensor
        fan_in = shape[0]
        # Initialize weights with a normal distribution scaled by sqrt(2/fan_in)
        return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)

class GlorotInitializer(Initializer):
    """
    Glorot (or Xavier) initialization for weights.
    
    This initialization is designed to maintain the same variance of the 
    activations and gradients across layers. It's well-suited for layers
    with sigmoid or tanh activations.
    
    Reference: Glorot & Bengio, "Understanding the difficulty of training
    deep feedforward neural networks", 2010.
    """
    def initialize(self, shape):
        """
        Initialize weights using Glorot initialization.
        
        Args:
            shape: The shape of the weights to initialize
            
        Returns:
            Initialized weights of the specified shape
        """
        # fan_in is the number of input units in the weight tensor
        # fan_out is the number of output units in the weight tensor
        fan_in, fan_out = shape[0], shape[1]
        # Initialize weights with a normal distribution scaled by sqrt(2/(fan_in + fan_out))
        return np.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))

# Function to apply initializer to a network
def apply_initializer(network, initializer):
    """
    Apply an initializer to all weight matrices in a network.
    
    Args:
        network: A NeuralNetwork object
        initializer: An Initializer object
    """
    for layer in network.layers:
        if hasattr(layer, 'params') and 'W' in layer.params:
            if hasattr(layer, 'input_channels'):  # Conv2D layer
                # Reshape convolutional weights for initialization
                original_shape = layer.params['W'].shape
                # Flatten the kernel dimensions for initialization
                flat_shape = (original_shape[0], np.prod(original_shape[1:]))
                # Initialize and reshape back
                layer.params['W'] = initializer.initialize(flat_shape).reshape(original_shape)
            else:  # Dense layer
                layer.params['W'] = initializer.initialize((layer.params['W'].shape[0], 
                                                         layer.params['W'].shape[1]))
