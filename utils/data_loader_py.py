"""
Data loading utilities for MNIST dataset.

This module provides functions for loading and preprocessing the MNIST dataset
of handwritten digits, preparing it for training and evaluation.
"""

import numpy as np
import os
import urllib.request
import gzip
import struct

def download_mnist(path="data"):
    """
    Download MNIST dataset if not already present.
    
    Args:
        path: Directory where the dataset will be stored
        
    Returns:
        Path to the dataset files
    """
    # URLs for the MNIST dataset files
    urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Download files if they don't exist
    for filename, url in urls.items():
        file_path = os.path.join(path, f"{filename}.gz")
        
        if not os.path.exists(file_path):
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {filename} to {file_path}")
    
    return path

def load_mnist_images(path, filename):
    """
    Load MNIST images from file.
    
    Args:
        path: Directory containing the dataset files
        filename: Name of the file containing images
        
    Returns:
        Numpy array of images
    """
    file_path = os.path.join(path, f"{filename}.gz")
    
    # Open and decompress the file
    with gzip.open(file_path, 'rb') as f:
        # Read file header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        # Read image data
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        
        # Reshape to 3D array: (num_images, rows, cols)
        images = images.reshape(num_images, rows, cols)
    
    return images

def load_mnist_labels(path, filename):
    """
    Load MNIST labels from file.
    
    Args:
        path: Directory containing the dataset files
        filename: Name of the file containing labels
        
    Returns:
        Numpy array of labels
    """
    file_path = os.path.join(path, f"{filename}.gz")
    
    # Open and decompress the file
    with gzip.open(file_path, 'rb') as f:
        # Read file header
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        # Read label data
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
    
    return labels

def load_data(path="data", flatten=False, normalize=True, validation_split=0.1, seed=42):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        path: Directory where the dataset is stored
        flatten: Whether to flatten images to 1D arrays
        normalize: Whether to normalize pixel values to [0, 1]
        validation_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Download dataset if necessary
    path = download_mnist(path)
    
    # Load data
    train_images = load_mnist_images(path, "train_images")
    train_labels = load_mnist_labels(path, "train_labels")
    test_images = load_mnist_images(path, "test_images")
    test_labels = load_mnist_labels(path, "test_labels")
    
    # Normalize pixel values to [0, 1]
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
    
    # Split training data into training and validation sets
    np.random.seed(seed)
    num_train = len(train_images)
    indices = np.random.permutation(num_train)
    num_val = int(num_train * validation_split)
    
    train_idx = indices[num_val:]
    val_idx = indices[:num_val]
    
    X_train = train_images[train_idx]
    y_train = train_labels[train_idx]
    X_val = train_images[val_idx]
    y_val = train_labels[val_idx]
    X_test = test_images
    y_test = test_labels
    
    # Reshape data for CNN or MLP
    if flatten:
        # Flatten for MLP: (samples, 28*28)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        # Reshape for CNN: (samples, channels, height, width)
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_val = X_val.reshape(X_val.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    
    print(f"Data loaded and preprocessed:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot_encode(labels, num_classes=10):
    """
    Convert labels to one-hot encoded format.
    
    Args:
        labels: Array of integer class labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def get_statistical_embeddings(images, flatten=True):
    """
    Generate statistical embeddings for images.
    
    This function extracts statistical features from images that can be
    useful for model interpretability.
    
    Args:
        images: Images array with shape (samples, height, width) or (samples, 1, height, width)
        flatten: Whether to flatten the output embeddings
        
    Returns:
        Statistical embeddings
    """
    # Ensure images are in shape (samples, height, width)
    if len(images.shape) == 4:
        if images.shape[1] == 1:
            images = images.reshape(images.shape[0], images.shape[2], images.shape[3])
    
    num_samples = images.shape[0]
    
    # Initialize embeddings
    embeddings = []
    
    for i in range(num_samples):
        img = images[i]
        
        # Calculate statistical features
        features = []
        
        # Mean and standard deviation of the whole image
        features.append(np.mean(img))
        features.append(np.std(img))
        
        # Mean and std for each row
        row_means = np.mean(img, axis=1)
        row_stds = np.std(img, axis=1)
        features.extend(row_means)
        features.extend(row_stds)
        
        # Mean and std for each column
        col_means = np.mean(img, axis=0)
        col_stds = np.std(img, axis=0)
        features.extend(col_means)
        features.extend(col_stds)
        
        # Quadrant features
        half_h = img.shape[0] // 2
        half_w = img.shape[1] // 2
        
        # Mean and std for each quadrant
        q1 = img[:half_h, :half_w]
        q2 = img[:half_h, half_w:]
        q3 = img[half_h:, :half_w]
        q4 = img[half_h:, half_w:]
        
        for q in [q1, q2, q3, q4]:
            features.append(np.mean(q))
            features.append(np.std(q))
            features.append(np.max(q))
            features.append(np.min(q))
        
        embeddings.append(features)
    
    embeddings = np.array(embeddings)
    
    if not flatten:
        # Reshape to a more structured format if needed
        num_features = embeddings.shape[1]
        embeddings = embeddings.reshape(num_samples, -1, 1)
    
    return embeddings

def create_batches(X, y, batch_size):
    """
    Create mini-batches from data.
    
    Args:
        X: Input data
        y: Target data
        batch_size: Batch size
        
    Returns:
        Generator yielding (X_batch, y_batch) tuples
    """
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
