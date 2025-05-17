"""
Implementation of the core Neural Network class.

This module provides a flexible framework for assembling layers into a complete
neural network, with support for training and inference.
"""

import numpy as np
from .layers import Dropout, BatchNormalization

class NeuralNetwork:
    """
    Neural Network class that manages layers, training, and inference.
    
    This class provides a high-level API for building and training neural networks
    with the layers, activations, optimizers, and loss functions defined in the
    neural_network package.
    """
    def __init__(self):
        """Initialize an empty neural network."""
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    def add(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: A layer object to add to the network
        """
        self.layers.append(layer)
    
    def set_loss(self, loss):
        """
        Set the loss function for the network.
        
        Args:
            loss: A loss object to use for computing training loss
        """
        self.loss = loss
    
    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the network.
        
        Args:
            optimizer: An optimizer object to use for updating parameters
        """
        self.optimizer = optimizer
    
    def forward(self, inputs, training=True):
        """
        Perform a forward pass through the network.
        
        Args:
            inputs: Input data for the network
            training: Boolean indicating whether in training mode
            
        Returns:
            Outputs from the final layer of the network
        """
        # Start with the input
        x = inputs
        
        # Forward pass through all layers
        for layer in self.layers:
            # Handle special layers that behave differently during training vs inference
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        
        return x
    
    def backward(self, dout=None):
        """
        Perform a backward pass through the network to compute gradients.
        
        Args:
            dout: Gradient of loss with respect to the output of the network.
                  If None, get the gradient from the loss function.
        """
        # If gradient is not provided, start backpropagation from the loss
        if dout is None:
            dout = self.loss.backward()
        
        # Backward pass through all layers in reverse order
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def update_params(self):
        """
        Update the parameters of all layers using the optimizer.
        """
        for layer in self.layers:
            # Only update layers with parameters
            if hasattr(layer, 'params') and layer.params:
                self.optimizer.update_params(layer)
    
    def train_step(self, inputs, targets):
        """
        Perform a single training step (forward pass, backward pass, parameter update).
        
        Args:
            inputs: Input data for the network
            targets: Target values for computing loss
            
        Returns:
            loss_value: The computed loss for this batch
            predictions: The network's predictions for the inputs
        """
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate loss
        loss_value = self.loss.forward(predictions, targets)
        
        # Backward pass
        self.backward()
        
        # Update parameters
        self.update_params()
        
        return loss_value, predictions
    
    def evaluate(self, inputs, targets):
        """
        Evaluate the network on the given inputs and targets.
        
        Args:
            inputs: Input data for the network
            targets: Target values for computing loss
            
        Returns:
            loss_value: The computed loss
            accuracy: The accuracy of the predictions
        """
        # Forward pass (no training)
        predictions = self.forward(inputs, training=False)
        
        # Calculate loss
        loss_value = self.loss.forward(predictions, targets)
        
        # Calculate accuracy
        if len(targets.shape) == 2:  # One-hot encoded
            target_indices = np.argmax(targets, axis=1)
        else:
            target_indices = targets
        
        predictions_indices = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions_indices == target_indices)
        
        return loss_value, accuracy
    
    def save_weights(self, filename):
        """
        Save the network's weights to a file.
        
        Args:
            filename: Path to save the weights
        """
        weights = {}
        
        # Collect weights from all layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                for param_name, param in layer.params.items():
                    weights[f'layer_{i}_{param_name}'] = param
        
        # Save weights
        np.save(filename, weights)
    
    def load_weights(self, filename):
        """
        Load weights from a file into the network.
        
        Args:
            filename: Path to the saved weights
        """
        weights = np.load(filename, allow_pickle=True).item()
        
        # Load weights into each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                for param_name in layer.params:
                    layer.params[param_name] = weights[f'layer_{i}_{param_name}']
