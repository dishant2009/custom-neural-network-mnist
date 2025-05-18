"""
Script for hyperparameter tuning of neural network models on MNIST.

This script performs hyperparameter optimization using either Optuna or Ray Tune,
with optional distributed execution on SLURM clusters.
"""

import argparse
import numpy as np
import os
import json
from datetime import datetime

# Import custom modules
from neural_network.network import NeuralNetwork
from neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from neural_network.activations import ReLU, Softmax
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import SGD, Adam

from training.initializers import HeInitializer, GlorotInitializer, apply_initializer
from training.trainer import Trainer

from utils.data_loader import load_data, one_hot_encode

from tuning.optuna_tuner import OptunaTuner
from tuning.ray_tuner import RayTuner
from tuning.slurm_config import generate_slurm_script, submit_slurm_job, collect_results

from mlflow_tracking.experiment_tracker import MLflowTracker

def network_builder(hidden_sizes=None, dropout_rate=0.2, use_batch_norm=True, 
                   conv_channels=None, kernel_size=3, pool_size=2, architecture='mlp'):
    """
    Build a neural network with the specified hyperparameters.
    
    Args:
        hidden_sizes: List of hidden layer sizes for MLP
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        conv_channels: List of convolutional channel sizes for CNN
        kernel_size: Kernel size for convolutional layers
        pool_size: Pool size for max pooling layers
        architecture: Network architecture ('mlp' or 'cnn')
        
    Returns:
        NeuralNetwork model
    """
    network = NeuralNetwork()
    
    if architecture == 'mlp':
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
            
        # Input layer
        network.add(Dense(784, hidden_sizes[0]))
        network.add(ReLU())
        
        if use_batch_norm:
            network.add(BatchNormalization(hidden_sizes[0]))
            
        network.add(Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            network.add(Dense(hidden_sizes[i-1], hidden_sizes[i]))
            network.add(ReLU())
            
            if use_batch_norm:
                network.add(BatchNormalization(hidden_sizes[i]))
                
            network.add(Dropout(dropout_rate))
        
        # Output layer
        network.add(Dense(hidden_sizes[-1], 10))
        network.add(Softmax())
    else:  # CNN
        if conv_channels is None:
            conv_channels = [32, 64]
        
        # First convolutional block
        network.add(Conv2D(1, conv_channels[0], kernel_size=kernel_size, padding=1))
        network.add(ReLU())
        network.add(MaxPool2D(pool_size=pool_size, stride=2))
        
        # Additional convolutional blocks
        for i in range(1, len(conv_channels)):
            network.add(Conv2D(conv_channels[i-1], conv_channels[i], 
                             kernel_size=kernel_size, padding=1))
            network.add(ReLU())
            network.add(MaxPool2D(pool_size=pool_size, stride=2))
        
        # Calculate feature map size after pooling
        feature_size = 28 // (2 ** len(conv_channels))
        flattened_size = feature_size * feature_size * conv_channels[-1]
        
        # Fully connected layers
        network.add(Flatten())
        network.add(Dense(flattened_size, 128))
        network.add(ReLU())
        network.add(Dropout(dropout_rate))
        network.add(Dense(128, 10))
        network.add(Softmax())
    
    return network

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Tune hyperparameters for MNIST neural network')
    
    # Tuning method
    parser.add_argument('--method', type=str, default='optuna', choices=['optuna', 'ray'],
                        help='Hyperparameter tuning method')
    
    # General tuning parameters
    parser.add_argument('--architecture', type=str, default=None, choices=['mlp', 'cnn', 'both'],
                        help='Network architecture to tune (default: both)')
    parser.add_argument('--num_trials', type=int, default=100, 
                        help='Number of trials for Optuna or samples for Ray')
    parser.add_argument('--objective', type=str, default='val_acc', choices=['val_acc', 'val_loss'],
                        help='Metric to optimize')
    
    # Distributed execution
    parser.add_argument('--distributed', action='store_true', 
                        help='Enable distributed execution')
    parser.add_argument('--trial_id', type=int, default=None, 
                        help='Trial ID for SLURM array jobs')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of parallel workers')
    
    # Ray Tune specific
    parser.add_argument('--cpus_per_trial', type=float, default=1, 
                        help='CPUs per trial for Ray Tune')
    parser.add_argument('--gpus_per_trial', type=float, default=0.25, 
                        help='GPUs per trial for Ray Tune')
    
    # SLURM specific
    parser.add_argument('--slurm_submit', action='store_true', 
                        help='Submit SLURM job')
    parser.add_argument('--slurm_partition', type=str, default='gpu', 
                        help='SLURM partition')
    parser.add_argument('--slurm_time', type=str, default='12:00:00', 
                        help='SLURM job time limit')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='hyperparameter_tuning',
                        help='Directory for output files')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='MLflow experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"mnist_tuning_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize MLflow tracking
    tracker = MLflowTracker(args.experiment_name)
    
    # Log parameters
    tracker.start_run(f"hyperparameter_tuning_{args.method}")
    tracker.log_params(vars(args))
    
    # Determine architectures to tune
    architectures = []
    if args.architecture is None or args.architecture == 'both':
        architectures = ['mlp', 'cnn']
    else:
        architectures = [args.architecture]
    
    # Setup SLURM distributed execution if requested
    if args.distributed and args.slurm_submit:
        # Generate SLURM script
        script_path = generate_slurm_script(
            job_name=f"mnist_tune_{args.method}",
            output_dir=os.path.join(args.output_dir, "slurm_logs"),
            python_script="tune_hyperparams.py",
            num_trials=args.num_trials,
            time=args.slurm_time,
            partition=args.slurm_partition
        )
        
        # Add command-line arguments to script
        with open(script_path, 'a') as f:
            f.write(f"\n# Pass command-line arguments\n")
            f.write(f"python tune_hyperparams.py --method {args.method} ")
            f.write(f"--num_trials 1 --trial_id $SLURM_ARRAY_TASK_ID ")
            if args.architecture:
                f.write(f"--architecture {args.architecture} ")
            f.write(f"--output_dir {args.output_dir} ")
            f.write(f"--experiment_name {args.experiment_name}\n")
        
        # Submit SLURM job
        job_id = submit_slurm_job(script_path)
        
        if job_id:
            print(f"SLURM job submitted with ID: {job_id}")
            print(f"Monitor job with: squeue -j {job_id}")
            
            # Log job information
            tracker.log_params({
                'slurm_job_id': job_id,
                'slurm_script': script_path
            })
            
            # End MLflow run
            tracker.end_run()
            return
    
    # Process a single trial for SLURM array job
    if args.trial_id is not None:
        print(f"Processing trial {args.trial_id}...")
        
        # Set random seed based on trial ID for reproducibility
        np.random.seed(args.trial_id)
        
        # For single trial SLURM execution, pick one architecture if not specified
        if args.architecture is None:
            # Deterministically choose architecture based on trial ID
            architecture = 'mlp' if args.trial_id % 2 == 0 else 'cnn'
        else:
            architecture = architectures[0]
        
        # Load MNIST data for the selected architecture
        if architecture == 'mlp':
            # Flatten images for MLP
            X_train, y_train, X_val, y_val, _, _ = load_data(flatten=True)
        else:  # CNN
            # Keep image shape for CNN
            X_train, y_train, X_val, y_val, _, _ = load_data(flatten=False)
        
        # One-hot encode labels
        y_train_one_hot = one_hot_encode(y_train)
        y_val_one_hot = one_hot_encode(y_val)
        
        # Create tuner for the selected architecture
        if args.method == 'optuna':
            # Prepare for single Optuna trial
            tuner = OptunaTuner(
                network_builder=network_builder,
                train_data=(X_train, y_train_one_hot),
                val_data=(X_val, y_val_one_hot),
                objective_metric=args.objective,
                direction='maximize' if args.objective == 'val_acc' else 'minimize'
            )
            
            # Run a single trial
            import optuna
            study = optuna.create_study(direction='maximize' if args.objective == 'val_acc' else 'minimize')
            trial = study.ask()
            trial_value = tuner.objective(trial)
            study.tell(trial, trial_value)
            
            # Save result
            result = {
                'trial_id': args.trial_id,
                'architecture': architecture,
                'params': trial.params,
                'value': float(trial_value),
                'objective': args.objective
            }
            
            result_path = os.path.join(args.output_dir, f"trial_{args.trial_id}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Trial {args.trial_id} completed with {args.objective} = {trial_value}")
            print(f"Results saved to {result_path}")
        else:
            # Ray tune doesn't support single trial execution in the same way
            print("Single trial execution not supported for Ray Tune")
            return
        
        return
    
    # For non-distributed or Ray Tune distributed execution
    results = {}
    
    for architecture in architectures:
        print(f"\nTuning {architecture.upper()} architecture...")
        
        # Load MNIST data for the current architecture
        if architecture == 'mlp':
            # Flatten images for MLP
            X_train, y_train, X_val, y_val, _, _ = load_data(flatten=True)
        else:  # CNN
            # Keep image shape for CNN
            X_train, y_train, X_val, y_val, _, _ = load_data(flatten=False)
        
        # One-hot encode labels
        y_train_one_hot = one_hot_encode(y_train)
        y_val_one_hot = one_hot_encode(y_val)
        
        # Start MLflow run for this architecture
        tracker.start_run(f"{architecture}_tuning")
        
        # Perform hyperparameter tuning
        if args.method == 'optuna':
            # Optuna tuning
            tuner = OptunaTuner(
                network_builder=network_builder,
                train_data=(X_train, y_train_one_hot),
                val_data=(X_val, y_val_one_hot),
                objective_metric=args.objective,
                direction='maximize' if args.objective == 'val_acc' else 'minimize'
            )
            
            # Run optimization
            study = tuner.tune(n_trials=args.num_trials, n_jobs=args.num_workers)
            
            # Extract best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Get best network
            best_network, _ = tuner.get_best_trial_network(study)
            
            # Log results
            tracker.log_params(best_params)
            tracker.log_metrics({
                'best_' + args.objective: best_value
            })
            
            # Save best model
            model_path = os.path.join(args.output_dir, f"best_{architecture}_model.npy")
            best_network.save_weights(model_path)
            tracker.log_model(best_network, f"best_{architecture}_model")
            
            # Store results
            results[architecture] = {
                'best_params': best_params,
                'best_value': float(best_value),
                'model_path': model_path
            }
            
            print(f"Best {args.objective} for {architecture}: {best_value}")
            print(f"Best parameters for {architecture}:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
        else:
            # Ray Tune
            tuner = RayTuner(
                network_builder=network_builder,
                train_data=(X_train, y_train_one_hot),
                val_data=(X_val, y_val_one_hot),
                objective_metric=args.objective
            )
            
            # Run optimization
            best_config, best_value = tuner.tune(
                num_samples=args.num_trials,
                max_concurrent=args.num_workers,
                cpus_per_trial=args.cpus_per_trial,
                gpus_per_trial=args.gpus_per_trial
            )
            
            # Build best network
            best_network = tuner.build_best_network(best_config)
            
            # Log results
            tracker.log_params(best_config)
            tracker.log_metrics({
                'best_' + args.objective: best_value
            })
            
            # Save best model
            model_path = os.path.join(args.output_dir, f"best_{architecture}_model.npy")
            best_network.save_weights(model_path)
            tracker.log_model(best_network, f"best_{architecture}_model")
            
            # Store results
            results[architecture] = {
                'best_params': best_config,
                'best_value': float(best_value),
                'model_path': model_path
            }
            
            print(f"Best {args.objective} for {architecture}: {best_value}")
            print(f"Best parameters for {architecture}:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
        
        # End MLflow run for this architecture
        tracker.end_run()
    
    # Save combined results
    results_path = os.path.join(args.output_dir, "tuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Start final summary run
    tracker.start_run("tuning_summary")
    
    # Log summary metrics
    for arch, result in results.items():
        tracker.log_metrics({
            f'{arch}_best_{args.objective}': result['best_value']
        })
    
    # Determine overall best architecture
    if len(architectures) > 1:
        best_arch = max(results.items(), key=lambda x: x[1]['best_value'] if args.objective == 'val_acc' else -x[1]['best_value'])[0]
        
        tracker.log_params({
            'best_architecture': best_arch,
            'best_model_path': results[best_arch]['model_path']
        })
        
        print(f"\nOverall best architecture: {best_arch}")
        print(f"Best {args.objective}: {results[best_arch]['best_value']}")
    
    # Log the results file
    tracker.log_artifact(results_path)
    
    # End final MLflow run
    tracker.end_run()
    
    print(f"\nHyperparameter tuning completed successfully!")
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
