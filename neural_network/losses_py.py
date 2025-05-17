"""
Implementation of loss functions for neural networks.

Loss functions measure the difference between the predicted outputs of the neural network
and the true target values. They provide a signal for the network to learn from during
backpropagation.
"""

import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Abstract base class for all loss functions.
    
    A loss function computes a scalar value that quantifies the difference
    between predictions and targets, with lower values indicating better performance.
    """
    @abstractmethod
    def forward(self, predictions, targets):
        """
        Compute the loss value.
        
        Args:
            predictions: Predicted values from the model
            targets: True target values
            
        Returns:
            A scalar loss value
        """
        pass
    
    @abstractmethod
    def backward(self):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Returns:
            Gradient of the loss with respect to the predictions
        """
        pass

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss implementation.
    
    This loss is commonly used for multi-class classification problems where
    each sample belongs to exactly one class. It's typically used with softmax
    activation in the output layer.
    """
    def forward(self, predictions, targets):
        """
        Compute the categorical cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities of shape (batch_size, num_classes)
            targets: True targets, either as one-hot encoded vectors of shape (batch_size, num_classes)
                     or as class indices of shape (batch_size,)
            
        Returns:
            A scalar loss value averaged over the batch
        """
        self.predictions = predictions
        self.targets = targets
        
        # Clip predictions to avoid log(0) which is undefined
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # If targets are one-hot encoded (e.g., [0, 1, 0, 0])
        if len(targets.shape) == 2:
            # For each sample, compute -sum(target * log(prediction))
            losses = -np.sum(targets * np.log(predictions_clipped), axis=1)
        # If targets are sparse (class indices, e.g., 1)
        else:
            # For each sample, compute -log(prediction) at the target index
            losses = -np.log(predictions_clipped[range(len(predictions)), targets])
        
        # Return mean loss over the batch
        return np.mean(losses)
    
    def backward(self):
        """
        Compute the gradient of categorical cross-entropy with respect to the predictions.
        
        Returns:
            Gradient of shape (batch_size, num_classes)
        """
        # Number of samples in the batch
        batch_size = len(self.predictions)
        
        # If targets are one-hot encoded
        if len(self.targets.shape) == 2:
            # Gradient is -target / prediction
            dinputs = -self.targets / self.predictions
        # If targets are sparse (class indices)
        else:
            # Initialize gradient array
            dinputs = np.zeros_like(self.predictions)
            # For each sample, set the gradient at the target index to -1/prediction
            dinputs[range(batch_size), self.targets] = -1 / self.predictions[range(batch_size), self.targets]
        
        # Normalize gradient by batch size
        dinputs = dinputs / batch_size
        
        return dinputs

class MeanSquaredError(Loss):
    """
    Mean Squared Error loss implementation.
    
    This loss measures the average of the squares of the errors—the average squared 
    difference between the estimated values and the actual value. It's commonly used 
    for regression problems.
    """
    def forward(self, predictions, targets):
        """
        Compute the mean squared error loss.
        
        Args:
            predictions: Predicted values of shape (batch_size, output_size)
            targets: True targets of shape (batch_size, output_size)
            
        Returns:
            A scalar loss value averaged over the batch
        """
        self.predictions = predictions
        self.targets = targets
        
        # Compute mean of squared differences
        return np.mean(np.square(predictions - targets))
    
    def backward(self):
        """
        Compute the gradient of mean squared error with respect to the predictions.
        
        Returns:
            Gradient of shape (batch_size, output_size)
        """
        # Number of samples in the batch
        batch_size = len(self.predictions)
        
        # Gradient is 2 * (prediction - target) / batch_size
        dinputs = 2 * (self.predictions - self.targets) / batch_size
        
        return dinputs
