"""
Visualization utilities for neural networks.

This module provides functions for visualizing training metrics, model predictions,
and other aspects of neural network training and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import itertools

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    return fig

def plot_mnist_images(images, labels=None, predictions=None, num_images=10, save_path=None):
    """
    Plot MNIST images with optional labels and predictions.
    
    Args:
        images: Array of images
        labels: Array of true labels (optional)
        predictions: Array of predicted labels (optional)
        num_images: Number of images to plot
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Limit to specified number of images
    num_images = min(num_images, len(images))
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
    
    # Ensure axes is a list even for a single image
    if num_images == 1:
        axes = [axes]
    
    # Plot each image
    for i in range(num_images):
        # Get the image
        img = images[i]
        
        # Reshape image if needed
        if len(img.shape) == 3 and img.shape[0] == 1:  # (1, height, width)
            img = img[0]
        elif len(img.shape) == 1:  # Flattened (784,)
            img = img.reshape(28, 28)
        
        # Plot the image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Add title with label and/or prediction
        title = ""
        if labels is not None:
            label = labels[i] if len(labels.shape) == 1 else np.argmax(labels[i])
            title += f"True: {label}"
        
        if predictions is not None:
            pred = predictions[i] if len(predictions.shape) == 1 else np.argmax(predictions[i])
            if title:
                title += f"\nPred: {pred}"
            else:
                title += f"Pred: {pred}"
        
        if title:
            axes[i].set_title(title)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"MNIST images plot saved to {save_path}")
    
    return fig

def plot_confusion_matrix(true_labels, predictions, class_names=None, normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        class_names: List of class names (optional)
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to class indices if one-hot encoded
    if len(true_labels.shape) > 1:
        true_labels = np.argmax(true_labels, axis=1)
    
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    num_classes = max(np.max(true_labels), np.max(predictions)) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(true_labels, predictions):
        cm[t, p] += 1
    
    # Normalize if requested
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values to cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    return plt.gcf()

def plot_feature_map(feature_map, title=None, save_path=None):
    """
    Plot a feature map from a convolutional layer.
    
    Args:
        feature_map: Feature map tensor of shape (channels, height, width)
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get number of channels
    num_channels = feature_map.shape[0]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Plot each channel
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(feature_map[i], cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_channels, grid_size * grid_size):
        axes[i].axis('off')
    
    # Add overall title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Feature map plot saved to {save_path}")
    
    return fig

def plot_weight_distribution(model, layer_indices=None, save_path=None):
    """
    Plot the distribution of weights in specified layers.
    
    Args:
        model: Neural network model
        layer_indices: List of layer indices to plot (optional, defaults to all layers with weights)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get layers with weights
    weight_layers = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'params') and 'W' in layer.params:
            weight_layers.append((i, layer))
    
    # If no layer indices specified, use all weight layers
    if layer_indices is None:
        layer_indices = [i for i, _ in weight_layers]
    
    # Filter for specified layers
    layers_to_plot = [(i, layer) for i, layer in weight_layers if i in layer_indices]
    
    if not layers_to_plot:
        print("No layers found with weights")
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(10, 3 * len(layers_to_plot)))
    
    # Ensure axes is a list even for a single layer
    if len(layers_to_plot) == 1:
        axes = [axes]
    
    # Plot weight distribution for each layer
    for ax, (i, layer) in zip(axes, layers_to_plot):
        weights = layer.params['W'].flatten()
        
        # Plot histogram
        ax.hist(weights, bins=50, alpha=0.7)
        ax.set_title(f'Layer {i} Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        mean = np.mean(weights)
        std = np.std(weights)
        ax.text(0.05, 0.95, f'Mean: {mean:.4f}\nStd: {std:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Weight distribution plot saved to {save_path}")
    
    return fig

def plot_learning_curve(train_sizes, train_scores, val_scores, title='Learning Curve', save_path=None):
    """
    Plot learning curve showing model performance vs training set size.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Array of training scores for each size
        val_scores: Array of validation scores for each size
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Calculate mean and std for scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training scores
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    # Plot validation scores
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='red')
    
    # Add labels and title
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curve plot saved to {save_path}")
    
    return plt.gcf()
