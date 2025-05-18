"""
Trainer class for managing the training process of neural networks.

This module provides a high-level interface for training neural networks,
handling the training loop, validation, and tracking of metrics.
"""

import numpy as np
from .mixed_precision import MixedPrecisionTraining

class Trainer:
    """
    Class to manage the training process of neural networks.
    
    The Trainer class provides a convenient interface for training neural networks,
    with support for mixed precision training, batch processing, and validation.
    """
    def __init__(self, network, optimizer, loss, batch_size=64, use_mixed_precision=False):
        """
        Initialize the trainer.
        
        Args:
            network: A NeuralNetwork object
            optimizer: An Optimizer object for parameter updates
            loss: A Loss object for computing training loss
            batch_size: Number of samples per batch
            use_mixed_precision: Whether to use mixed precision training
        """
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Set network loss and optimizer
        self.network.set_loss(loss)
        self.network.set_optimizer(optimizer)
        
        # Setup mixed precision training if needed
        if self.use_mixed_precision:
            self.mp_trainer = MixedPrecisionTraining(network, optimizer)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, lr_scheduler=None, verbose=True):
        """
        Train the network on the given data.
        
        Args:
            X_train: Training inputs
            y_train: Training targets
            X_val: Validation inputs
            y_val: Validation targets
            epochs: Number of training epochs
            lr_scheduler: Learning rate scheduler (optional)
            verbose: Whether to print progress information
            
        Returns:
            history: Dictionary containing training and validation metrics
        """
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Number of training samples and batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Initialize epoch metrics
            epoch_loss = 0
            epoch_acc = 0
            
            # Process each batch
            for batch in range(n_batches):
                # Get batch data
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Train on batch
                if self.use_mixed_precision:
                    batch_loss, predictions = self.mp_trainer.train_step(X_batch, y_batch)
                else:
                    batch_loss, predictions = self.network.train_step(X_batch, y_batch)
                
                # Calculate batch accuracy
                if len(y_batch.shape) == 2:  # One-hot encoded
                    y_true = np.argmax(y_batch, axis=1)
                else:
                    y_true = y_batch
                
                y_pred = np.argmax(predictions, axis=1)
                batch_acc = np.mean(y_pred == y_true)
                
                # Update epoch metrics (weighted by batch size)
                batch_weight = (end_idx - start_idx) / n_samples
                epoch_loss += batch_loss * batch_weight
                epoch_acc += batch_acc * batch_weight
            
            # Evaluate on validation data
            val_loss, val_acc = self.network.evaluate(X_val, y_val)
            
            # Update history
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate if scheduler is provided
            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            
            # Print progress
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - '
                      f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        
        return history


class LearningRateScheduler:
    """
    Learning rate scheduler with support for various strategies.
    
    This scheduler can adjust the learning rate during training based on
    performance metrics or predefined schedules.
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6):
        """
        Initialize the learning rate scheduler.
        
        Args:
            optimizer: An Optimizer object
            mode: 'min' or 'max' depending on whether we want to minimize or maximize the monitored quantity
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which learning rate will be reduced
            min_lr: Lower bound on the learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
    
    def step(self, metric):
        """
        Update the learning rate based on the performance metric.
        
        Args:
            metric: Performance metric to monitor (e.g., validation loss)
        """
        if self.mode == 'min':
            is_better = metric < self.best
        else:
            is_better = metric > self.best
        
        if is_better:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                self.optimizer.learning_rate = max(self.optimizer.learning_rate * self.factor, self.min_lr)
                self.wait = 0
                print(f'Reducing learning rate to {self.optimizer.learning_rate:.6f}')
