"""
Hyperparameter tuning using Optuna.

Optuna is a hyperparameter optimization framework that provides efficient search
algorithms for finding the best hyperparameters for machine learning models.
"""

import optuna
import numpy as np
import tempfile
import joblib
import os

from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import SGD, Adam
from training.initializers import HeInitializer, GlorotInitializer, apply_initializer
from training.trainer import Trainer

class OptunaTuner:
    """
    Hyperparameter tuner using Optuna.
    
    This class provides methods for optimizing hyperparameters of neural networks
    using Optuna's Bayesian optimization methods.
    """
    def __init__(self, network_builder, train_data, val_data, objective_metric='val_acc', direction='maximize'):
        """
        Initialize the Optuna tuner.
        
        Args:
            network_builder: Function that builds a network with given hyperparameters
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            objective_metric: Metric to optimize ('val_acc' or 'val_loss')
            direction: Direction of optimization ('maximize' or 'minimize')
        """
        self.network_builder = network_builder
        self.train_data = train_data
        self.val_data = val_data
        self.objective_metric = objective_metric
        self.direction = direction
        
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Value of the objective metric for this trial
        """
        # Define hyperparameters to tune
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
        
        # Architecture selection
        architecture = trial.suggest_categorical('architecture', ['mlp', 'cnn'])
        
        # MLP specific params
        if architecture == 'mlp':
            n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)
            hidden_sizes = []
            
            for i in range(n_hidden_layers):
                hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 32, 512))
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            
            # Build network
            network = self.network_builder(
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                architecture='mlp'
            )
        else:  # CNN
            n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
            conv_channels = []
            
            for i in range(n_conv_layers):
                conv_channels.append(trial.suggest_int(f'conv_channels_{i}', 16, 128))
            
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
            pool_size = trial.suggest_categorical('pool_size', [2])
            
            # Build network
            network = self.network_builder(
                conv_channels=conv_channels,
                kernel_size=kernel_size,
                pool_size=pool_size,
                architecture='cnn'
            )
        
        # Initializer
        initializer_type = trial.suggest_categorical('initializer', ['he', 'glorot'])
        if initializer_type == 'he':
            initializer = HeInitializer()
        else:
            initializer = GlorotInitializer()
        
        apply_initializer(network, initializer)
        
        # Optimizer with gradient clipping
        clip_value = trial.suggest_float('clip_value', 1.0, 10.0)
        
        if optimizer_type == 'sgd':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            optimizer = SGD(learning_rate=lr, momentum=momentum, clip_value=clip_value)
        else:
            beta1 = trial.suggest_float('beta1', 0.85, 0.95)
            beta2 = trial.suggest_float('beta2', 0.99, 0.999)
            optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, clip_value=clip_value)
        
        # Loss
        loss = CategoricalCrossEntropy()
        
        # Use mixed precision?
        use_mixed_precision = trial.suggest_categorical('use_mixed_precision', [True, False])
        
        # Trainer
        trainer = Trainer(network, optimizer, loss, batch_size=batch_size, 
                          use_mixed_precision=use_mixed_precision)
        
        # Train for a few epochs for quick evaluation
        epochs = 5
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Suppress verbose output during tuning
        history = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=False)
        
        # Return objective metric
        if self.objective_metric == 'val_acc':
            return history['val_acc'][-1]
        elif self.objective_metric == 'val_loss':
            return history['val_loss'][-1]
    
    def tune(self, n_trials=100, n_jobs=1):
        """
        Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs
            
        Returns:
            Best trial object containing optimized hyperparameters
        """
        # Create study (optimization process)
        study = optuna.create_study(direction=self.direction)
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        
        # Print results
        print('\nHyperparameter Tuning Results:')
        print('=' * 50)
        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params:')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
        
        # Save study for later analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            joblib.dump(study, f)
            filename = f.name
        
        print(f'\nStudy saved to {filename}')
        
        return study
    
    def get_best_trial_network(self, study):
        """
        Build a network using the best trial parameters.
        
        Args:
            study: Optuna study object after optimization
            
        Returns:
            Best network based on the optimized hyperparameters
        """
        best_params = study.best_trial.params
        
        # Build network using best parameters
        if best_params['architecture'] == 'mlp':
            # Extract hidden sizes
            hidden_sizes = []
            for i in range(best_params['n_hidden_layers']):
                hidden_sizes.append(best_params[f'hidden_size_{i}'])
            
            network = self.network_builder(
                hidden_sizes=hidden_sizes,
                dropout_rate=best_params['dropout_rate'],
                use_batch_norm=best_params['use_batch_norm'],
                architecture='mlp'
            )
        else:  # CNN
            # Extract conv channels
            conv_channels = []
            for i in range(best_params['n_conv_layers']):
                conv_channels.append(best_params[f'conv_channels_{i}'])
            
            network = self.network_builder(
                conv_channels=conv_channels,
                kernel_size=best_params['kernel_size'],
                pool_size=best_params['pool_size'],
                architecture='cnn'
            )
        
        # Apply initializer
        if best_params['initializer'] == 'he':
            initializer = HeInitializer()
        else:
            initializer = GlorotInitializer()
        
        apply_initializer(network, initializer)
        
        return network, best_params
