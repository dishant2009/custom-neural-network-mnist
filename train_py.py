"""
Main script for training neural networks on MNIST.

This script trains either MLP or CNN models on the MNIST dataset,
with support for various optimization and regularization techniques.
"""

import argparse
import numpy as np
import os
import time
from datetime import datetime

# Import custom modules
from neural_network.network import NeuralNetwork
from neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from neural_network.activations import ReLU, Softmax
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import SGD, Adam

from training.initializers import HeInitializer, GlorotInitializer, apply_initializer
from training.trainer import Trainer, LearningRateScheduler

from utils.data_loader import load_data, one_hot_encode
from utils.visualization import plot_training_history, plot_mnist_images
from utils.metrics import evaluate_model

from mlflow_tracking.experiment_tracker import MLflowTracker

def build_mlp(input_size=784, hidden_sizes=[256, 128], output_size=10, 
             dropout_rate=0.2, use_batch_norm=True):
    """
    Build a Multilayer Perceptron (MLP) model.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output classes
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        NeuralNetwork model
    """
    network = NeuralNetwork()
    
    # Input layer
    network.add(Dense(input_size, hidden_sizes[0]))
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
    network.add(Dense(hidden_sizes[-1], output_size))
    network.add(Softmax())
    
    return network

def build_cnn(input_channels=1, conv_channels=[32, 64], fc_size=128, output_size=10,
             kernel_size=3, pool_size=2, dropout_rate=0.2):
    """
    Build a Convolutional Neural Network (CNN) model.
    
    Args:
        input_channels: Number of input channels
        conv_channels: List of convolutional layer channel sizes
        fc_size: Size of the fully connected layer
        output_size: Number of output classes
        kernel_size: Size of the convolutional kernels
        pool_size: Size of the pooling windows
        dropout_rate: Dropout rate for regularization
        
    Returns:
        NeuralNetwork model
    """
    network = NeuralNetwork()
    
    # First convolutional block
    network.add(Conv2D(input_channels, conv_channels[0], kernel_size=kernel_size, padding=1))
    network.add(ReLU())
    network.add(MaxPool2D(pool_size=pool_size, stride=2))
    
    # Additional convolutional blocks
    for i in range(1, len(conv_channels)):
        network.add(Conv2D(conv_channels[i-1], conv_channels[i], kernel_size=kernel_size, padding=1))
        network.add(ReLU())
        network.add(MaxPool2D(pool_size=pool_size, stride=2))
    
    # Flatten the feature maps
    network.add(Flatten())
    
    # Calculate the size of the flattened feature maps
    # For MNIST (28x28), after 2 pooling layers with stride 2, we get 7x7 feature maps
    feature_map_size = 28 // (2 ** len(conv_channels))
    flattened_size = feature_map_size * feature_map_size * conv_channels[-1]
    
    # Fully connected layers
    network.add(Dense(flattened_size, fc_size))
    network.add(ReLU())
    network.add(Dropout(dropout_rate))
    
    # Output layer
    network.add(Dense(fc_size, output_size))
    network.add(Softmax())
    
    return network

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    
    # Model architecture
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='Network architecture (mlp or cnn)')
    
    # MLP specific arguments
    parser.add_argument('--hidden_sizes', type=str, default='256,128',
                        help='Comma-separated list of hidden layer sizes (for MLP)')
    
    # CNN specific arguments
    parser.add_argument('--conv_channels', type=str, default='32,64',
                        help='Comma-separated list of convolutional channel sizes (for CNN)')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for convolutional layers (for CNN)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'],
                        help='Optimizer (sgd or adam)')
    
    # Regularization
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    
    # Initialization
    parser.add_argument('--initializer', type=str, default='he', choices=['he', 'glorot'],
                        help='Weight initializer')
    
    # Mixed precision
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='MLflow experiment name (default: auto-generated)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name (default: auto-generated)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for output files')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name for saved model (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse lists from command-line arguments
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]
    conv_channels = [int(channels) for channels in args.conv_channels.split(',')]
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"mnist_{args.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate model name if not provided
    if args.model_name is None:
        args.model_name = f"{args.architecture}_model_{int(time.time())}"
    
    # Initialize MLflow tracking
    tracker = MLflowTracker(args.experiment_name)
    tracker.start_run(args.run_name)
    
    # Log parameters
    tracker.log_params(vars(args))
    
    # Load MNIST data
    print("Loading MNIST data...")
    if args.architecture == 'mlp':
        # Flatten images for MLP
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(flatten=True)
    else:  # CNN
        # Keep image shape for CNN
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(flatten=False)
    
    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train)
    y_val_one_hot = one_hot_encode(y_val)
    y_test_one_hot = one_hot_encode(y_test)
    
    # Build network
    print(f"Building {args.architecture.upper()} model...")
    if args.architecture == 'mlp':
        network = build_mlp(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=10,
            dropout_rate=args.dropout_rate,
            use_batch_norm=args.use_batch_norm
        )
    else:  # CNN
        network = build_cnn(
            input_channels=X_train.shape[1],
            conv_channels=conv_channels,
            output_size=10,
            kernel_size=args.kernel_size,
            dropout_rate=args.dropout_rate
        )
    
    # Apply weight initialization
    if args.initializer == 'he':
        initializer = HeInitializer()
    else:  # glorot
        initializer = GlorotInitializer()
    
    apply_initializer(network, initializer)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9, clip_value=5.0)
    else:  # adam
        optimizer = Adam(learning_rate=args.learning_rate, clip_value=5.0)
    
    # Create loss function
    loss = CategoricalCrossEntropy()
    
    # Create learning rate scheduler
    lr_scheduler = LearningRateScheduler(optimizer, patience=3, factor=0.5)
    
    # Create trainer
    trainer = Trainer(
        network, optimizer, loss, 
        batch_size=args.batch_size,
        use_mixed_precision=args.mixed_precision
    )
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    history = trainer.train(
        X_train, y_train_one_hot, 
        X_val, y_val_one_hot, 
        epochs=args.epochs,
        lr_scheduler=lr_scheduler
    )
    
    # Log training history
    tracker.log_history(history)
    
    # Plot training history
    history_plot = plot_training_history(history)
    history_plot_path = os.path.join(args.output_dir, f"{args.model_name}_history.png")
    history_plot.savefig(history_plot_path)
    tracker.log_artifact(history_plot_path)
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(network, X_test, y_test_one_hot)
    
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    
    # Log test metrics
    tracker.log_metrics({
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'test_precision': test_metrics.get('precision', 0),
        'test_recall': test_metrics.get('recall', 0),
        'test_f1': test_metrics.get('f1_score', 0)
    })
    
    # Plot example predictions
    predictions = network.forward(X_test[:10], training=False)
    pred_plot = plot_mnist_images(X_test[:10], y_test[:10], predictions)
    pred_plot_path = os.path.join(args.output_dir, f"{args.model_name}_predictions.png")
    pred_plot.savefig(pred_plot_path)
    tracker.log_artifact(pred_plot_path)
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_name}.npy")
    network.save_weights(model_path)
    tracker.log_model(network, args.model_name)
    
    print(f"Model saved to {model_path}")
    
    # End MLflow run
    tracker.end_run()
    
    print("Training completed successfully!")
    return network, X_test, y_test

if __name__ == '__main__':
    main()
