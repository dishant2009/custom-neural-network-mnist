"""
Implementation of optimization algorithms for neural networks.

Optimizers update the model parameters based on the gradients computed during
backpropagation, with the goal of minimizing the loss function.
"""

import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    An optimizer is responsible for updating the parameters of a layer
    based on the gradients computed during backpropagation.
    """
    @abstractmethod
    def update_params(self, layer):
        """
        Update the parameters of a layer.
        
        Args:
            layer: A layer object with params and grads attributes
        """
        pass

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    
    Classic optimization algorithm that can be enhanced with momentum
    to accelerate convergence and help escape local minima.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0, clip_value=None):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            momentum: Momentum coefficient, helps accelerate SGD
            clip_value: Maximum allowed gradient value, helps with gradient explosion
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.clip_value = clip_value
        self.velocities = {}  # Store velocities for momentum computation
    
    def update_params(self, layer):
        """
        Update the parameters of a layer using SGD with momentum.
        
        Args:
            layer: A layer object with params and grads attributes
        """
        # Initialize velocities if not exists
        for param_name, param in layer.params.items():
            # Create a unique key for each parameter using the param name and layer id
            if param_name not in self.velocities:
                self.velocities[param_name + '_' + str(id(layer))] = np.zeros_like(param)
        
        # Apply gradient clipping if specified
        if self.clip_value is not None:
            for grad_name, grad in layer.grads.items():
                # Clip gradients to prevent explosion
                layer.grads[grad_name] = np.clip(grad, -self.clip_value, self.clip_value)
        
        # Update parameters with momentum
        for param_name, param in layer.params.items():
            velocity_key = param_name + '_' + str(id(layer))
            
            # Update velocity using momentum
            self.velocities[velocity_key] = (
                self.momentum * self.velocities[velocity_key] - 
                self.learning_rate * layer.grads[param_name]
            )
            
            # Update parameter using the velocity
            layer.params[param_name] += self.velocities[velocity_key]

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    An extension to stochastic gradient descent that combines the benefits of
    AdaGrad and RMSProp. It maintains per-parameter learning rates adapted 
    based on estimates of first and second moments of the gradients.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=None):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            beta1: Exponential decay rate for the first moment estimates
            beta2: Exponential decay rate for the second moment estimates
            epsilon: Small constant for numerical stability
            clip_value: Maximum allowed gradient value, helps with gradient explosion
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
    
    def update_params(self, layer):
        """
        Update the parameters of a layer using Adam.
        
        Args:
            layer: A layer object with params and grads attributes
        """
        # Increment time step
        self.t += 1
        
        # Initialize moment estimates if not exists
        for param_name, param in layer.params.items():
            key = param_name + '_' + str(id(layer))
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
        
        # Apply gradient clipping if specified
        if self.clip_value is not None:
            for grad_name, grad in layer.grads.items():
                layer.grads[grad_name] = np.clip(grad, -self.clip_value, self.clip_value)
        
        # Update parameters using Adam
        for param_name, param in layer.params.items():
            key = param_name + '_' + str(id(layer))
            
            # Update biased first moment estimate (momentum)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * layer.grads[param_name]
            
            # Update biased second raw moment estimate (RMSProp)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(layer.grads[param_name])
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)  # Correct bias in first moment
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)  # Correct bias in second moment
            
            # Update parameters with adaptive learning rate
            layer.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
