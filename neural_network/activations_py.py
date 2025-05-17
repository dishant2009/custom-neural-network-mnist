"""
Implementation of various activation functions for neural networks.

This module provides activation functions that introduce non-linearity to neural networks,
which is essential for learning complex patterns. Each activation implements forward and
backward passes required for training with backpropagation.
"""

import numpy as np
from .layers import Layer

class Activation(Layer):
    """
    Base class for all activation functions.
    
    Activation functions are a special type of layer that applies a non-linear 
    transformation to its input.
    """
    def __init__(self):
        super().__init__()
    
class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    
    ReLU is one of the most widely used activation functions in deep learning.
    It returns x if x > 0, else 0. This simple non-linearity helps with vanishing
    gradient problems in deep networks.
    """
    def forward(self, inputs):
        """
        Forward pass for ReLU activation.
        
        Args:
            inputs: Input values of any shape
            
        Returns:
            outputs: max(0, inputs) element-wise
        """
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, dout):
        """
        Backward pass for ReLU activation.
        
        Args:
            dout: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # Gradient is 1 where input was > 0, and 0 elsewhere
        return dout * (self.inputs > 0)

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    The sigmoid function transforms inputs to a range between 0 and 1,
    making it useful for models where the output needs to be interpreted
    as a probability.
    """
    def forward(self, inputs):
        """
        Forward pass for sigmoid activation.
        
        Args:
            inputs: Input values of any shape
            
        Returns:
            outputs: 1 / (1 + exp(-inputs)) element-wise
        """
        # Store outputs for use in backward pass
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def backward(self, dout):
        """
        Backward pass for sigmoid activation.
        
        Args:
            dout: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        return dout * self.outputs * (1 - self.outputs)

class Softmax(Activation):
    """
    Softmax activation function.
    
    The softmax function turns a vector of K real values into a probability
    distribution of K possible outcomes. It's often used in the final layer
    of a classification network.
    """
    def forward(self, inputs):
        """
        Forward pass for softmax activation.
        
        Args:
            inputs: Input values of shape (batch_size, num_classes)
            
        Returns:
            outputs: Softmax probabilities of shape (batch_size, num_classes)
        """
        # Shift inputs for numerical stability (prevents overflow)
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.outputs
    
    def backward(self, dout):
        """
        Backward pass for softmax activation.
        
        Args:
            dout: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # When used with cross-entropy loss, the gradient simplifies to (output - target)
        # This is already handled in the cross-entropy loss backward pass
        return dout
