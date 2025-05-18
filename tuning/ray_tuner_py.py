"""
Distributed hyperparameter tuning using Ray Tune.

Ray Tune provides efficient distributed hyperparameter tuning with support for
early stopping and various search algorithms. It enables parallel evaluation
of hyperparameters across multiple machines and cores.
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np

from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import SGD, Adam
from training.initializers import HeInitializer, GlorotInitializer, apply_initializer
from training.trainer import Trainer

class RayTuner:
    """
    Hyperparameter tuner using Ray Tune for distributed optimization.
    
    This class provides methods for optimizing hyperparameters of neural networks
    using Ray Tune's asynchronous optimization methods across multiple cores or machines.
    """
    def __init__(self, network_builder, train_data, val_data, objective_metric='val_acc'):
        """
        Initialize the Ray tuner.
        
        Args:
            network_builder: Function that builds a network with given hyperparameters
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            objective_metric: Metric to optimize ('val_acc' or 'val_loss')
        """
        self.network_builder = network_builder
        self.train_data = train_data
        self.val_data = val_data
        self.objective_metric = objective_metric
    
    def train_func(self, config):
        """
        Training function for Ray Tune.
        
        This function builds and trains a network using the hyperparameters
        specified in config, and reports metrics back to Ray Tune.
        
        Args:
            config: Dictionary of hyperparameters
        """
        # Build network based on architecture
        if config['architecture'] == 'mlp':
            network = self.network_builder(
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout_rate'],
                use_batch_norm=config['use_batch_norm'],
                architecture='mlp'
            )
        else:  # CNN
            network = self.network_builder(
                conv_channels=config['conv_channels'],
                kernel_size=config['kernel_size'],
                pool_size=config['pool_size'],
                architecture='cnn'
            )
        
        # Apply initializer
        if config['initializer'] == 'he':
            initializer = HeInitializer()
        else:
            initializer = GlorotInitializer()
        
        apply_initializer(network, initializer)
        
        # Configure optimizer
        if config['optimizer'] == 'sgd':
            optimizer = SGD(
                learning_rate=config['learning_rate'],
                momentum=config['momentum'],
                clip_value=config['clip_value']
            )
        else:
            optimizer = Adam(
                learning_rate=config['learning_rate'],
                beta1=config['beta1'],
                beta2=config['beta2'],
                clip_value=config['clip_value']
            )
        
        # Loss function
        loss = CategoricalCrossEntropy()
        
        # Setup trainer
        trainer = Trainer(
            network, optimizer, loss, 
            batch_size=config['batch_size'],
            use_mixed_precision=config['use_mixed_precision']
        )
        
        # Get data
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Train for specified epochs, reporting metrics after each epoch
        for epoch in range(config['epochs']):
            # Train for one epoch
            history = trainer.train(X_train, y_train, X_val, y_val, epochs=1, verbose=False)
            
            # Report metrics to Ray Tune
            metrics = {
                'train_loss': history['train_loss'][-1],
                'train_acc': history['train_acc'][-1],
                'val_loss': history['val_loss'][-1],
                'val_acc': history['val_acc'][-1]
            }
            
            # Report the requested metric for optimization
            tune.report(**metrics)
    
    def get_config_space(self):
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary specifying the search space for hyperparameters
        """
        config = {
            # Common hyperparameters
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'batch_size': tune.choice([16, 32, 64, 128, 256]),
            'optimizer': tune.choice(['sgd', 'adam']),
            'clip_value': tune.uniform(1.0, 10.0),
            'architecture': tune.choice(['mlp', 'cnn']),
            'initializer': tune.choice(['he', 'glorot']),
            'use_mixed_precision': tune.choice([True, False]),
            'epochs': 5,  # Number of epochs per trial
            
            # MLP specific hyperparameters
            'hidden_sizes': tune.sample_from(
                lambda _: [tune.randint(32, 512).sample() for _ in range(tune.randint(1, 3).sample())]
            ),
            'dropout_rate': tune.uniform(0.0, 0.5),
            'use_batch_norm': tune.choice([True, False]),
            
            # CNN specific hyperparameters
            'conv_channels': tune.sample_from(
                lambda _: [tune.randint(16, 128).sample() for _ in range(tune.randint(1, 3).sample())]
            ),
            'kernel_size': tune.choice([3, 5]),
            'pool_size': tune.choice([2]),
            
            # Optimizer specific hyperparameters
            'momentum': tune.uniform(0.0, 0.99),  # For SGD
            'beta1': tune.uniform(0.85, 0.95),    # For Adam
            'beta2': tune.uniform(0.99, 0.999),   # For Adam
        }
        
        return config
    
    def tune(self, num_samples=10, max_concurrent=4, cpus_per_trial=1, gpus_per_trial=0.25):
        """
        Run distributed hyperparameter tuning.
        
        Args:
            num_samples: Number of hyperparameter combinations to try
            max_concurrent: Maximum number of concurrent trials
            cpus_per_trial: CPUs allocated per trial
            gpus_per_trial: GPUs allocated per trial (fractional values supported)
            
        Returns:
            Tuple of (best_config, best_metric_value)
        """
        # Initialize Ray
        ray.init()
        
        # Configure the search space
        config = self.get_config_space()
        
        # Configure the search algorithm (ASHA scheduler for efficient early stopping)
        scheduler = ASHAScheduler(
            max_t=5,                # Max epochs
            grace_period=1,         # Minimum epochs before stopping
            reduction_factor=2      # Reduction factor for bracketing
        )
        
        print("\nStarting distributed hyperparameter tuning with Ray Tune")
        print("=" * 70)
        print(f"Running {num_samples} trials with up to {max_concurrent} concurrent trials")
        
        # Run the hyperparameter search
        result = tune.run(
            self.train_func,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
            metric=self.objective_metric,
            mode='max' if self.objective_metric == 'val_acc' else 'min',
            max_concurrent_trials=max_concurrent,
            verbose=1
        )
        
        # Get the best trial
        best_trial = result.get_best_trial(
            self.objective_metric, 
            'max' if self.objective_metric == 'val_acc' else 'min', 
            'last'  # Get the last value of the metric
        )
        
        # Get the best config and value
        best_config = best_trial.config
        best_value = best_trial.last_result[self.objective_metric]
        
        print("\nHyperparameter tuning completed!")
        print("=" * 70)
        print(f"Best {self.objective_metric}: {best_value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        # Shutdown Ray
        ray.shutdown()
        
        return best_config, best_value
    
    def build_best_network(self, best_config):
        """
        Build a network using the best found hyperparameters.
        
        Args:
            best_config: Dictionary of best hyperparameters
            
        Returns:
            Network built with the best hyperparameters
        """
        # Build network based on architecture
        if best_config['architecture'] == 'mlp':
            network = self.network_builder(
                hidden_sizes=best_config['hidden_sizes'],
                dropout_rate=best_config['dropout_rate'],
                use_batch_norm=best_config['use_batch_norm'],
                architecture='mlp'
            )
        else:  # CNN
            network = self.network_builder(
                conv_channels=best_config['conv_channels'],
                kernel_size=best_config['kernel_size'],
                pool_size=best_config['pool_size'],
                architecture='cnn'
            )
        
        # Apply initializer
        if best_config['initializer'] == 'he':
            initializer = HeInitializer()
        else:
            initializer = GlorotInitializer()
        
        apply_initializer(network, initializer)
        
        return network
