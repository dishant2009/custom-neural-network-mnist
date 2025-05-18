"""
Mixed precision training support for neural networks.

Mixed precision training uses lower precision formats (like float16) where possible
to reduce memory usage and increase computation speed, while maintaining model accuracy
with carefully managed numeric stability techniques.
"""

import numpy as np

class MixedPrecisionTraining:
    """
    Implementation of mixed precision training.
    
    Mixed precision training uses FP16 (half-precision) for most operations
    to increase computational efficiency, while keeping a master copy of weights 
    in FP32 (single-precision) for numeric stability.
    
    Reference: Micikevicius et al., "Mixed Precision Training", 2018.
    """
    def __init__(self, network, optimizer, loss_scale=128.0):
        """
        Initialize mixed precision training.
        
        Args:
            network: A NeuralNetwork object
            optimizer: An Optimizer object
            loss_scale: Scaling factor for the loss to prevent underflow in FP16
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_scale = loss_scale
    
    def to_fp16(self, network):
        """
        Convert network parameters to FP16 (half precision).
        
        Args:
            network: A NeuralNetwork object
        """
        for layer in network.layers:
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    # Convert parameters to half precision
                    layer.params[param_name] = param.astype(np.float16)
    
    def to_fp32(self, network):
        """
        Convert network parameters to FP32 (single precision).
        
        Args:
            network: A NeuralNetwork object
        """
        for layer in network.layers:
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    # Convert parameters to single precision
                    layer.params[param_name] = param.astype(np.float32)
    
    def train_step(self, inputs, targets):
        """
        Perform a mixed precision training step.
        
        Args:
            inputs: Input data for the network
            targets: Target values for computing loss
            
        Returns:
            loss_value: The computed loss for this batch
            predictions: The network's predictions for the inputs
        """
        # Convert to FP16 for forward pass
        self.to_fp16(self.network)
        
        # Forward pass in FP16
        predictions = self.network.forward(inputs)
        
        # Calculate loss
        loss_value = self.network.loss.forward(predictions, targets)
        
        # Scale loss to prevent underflow in gradients
        scaled_loss = loss_value * self.loss_scale
        
        # Backward pass with scaled gradients
        self.network.backward()
        
        # Unscale gradients before weight update
        for layer in self.network.layers:
            if hasattr(layer, 'grads'):
                for grad_name, grad in layer.grads.items():
                    layer.grads[grad_name] = grad / self.loss_scale
        
        # Convert back to FP32 for optimizer update
        self.to_fp32(self.network)
        
        # Update parameters in FP32
        self.network.update_params()
        
        # Return unscaled loss and predictions
        return loss_value, predictions
