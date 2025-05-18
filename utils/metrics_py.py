"""
Metrics for evaluating neural network performance.

This module provides functions for calculating various metrics to evaluate
the performance of neural network models, particularly for classification tasks.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels (class indices or one-hot encoded)
        y_pred: Predicted labels or probabilities
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Convert one-hot encoded labels to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Convert probabilities to class indices
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    return np.mean(y_pred == y_true)

def precision_recall_f1(y_true, y_pred, average='macro'):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        y_true: True labels (class indices or one-hot encoded)
        y_pred: Predicted labels or probabilities
        average: Averaging method ('macro', 'micro', 'weighted', or None for per-class)
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Convert one-hot encoded labels to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Convert probabilities to class indices
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # True positives: diagonal elements
    tp = np.diag(cm)
    
    # False positives: column sum - true positives
    fp = np.sum(cm, axis=0) - tp
    
    # False negatives: row sum - true positives
    fn = np.sum(cm, axis=1) - tp
    
    # Calculate per-class precision and recall
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        # Replace NaN with 0
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
    
    # Average metrics if requested
    if average == 'macro':
        # Simple unweighted average
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
    elif average == 'micro':
        # Average using total counts
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)
        
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    elif average == 'weighted':
        # Average weighted by support
        support = np.sum(cm, axis=1)
        total_support = np.sum(support)
        
        precision = np.sum(precision * support) / total_support
        recall = np.sum(recall * support) / total_support
        f1 = np.sum(f1 * support) / total_support
    
    return precision, recall, f1

def confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate metrics from confusion matrix.
    
    Args:
        y_true: True labels (class indices or one-hot encoded)
        y_pred: Predicted labels or probabilities
        
    Returns:
        Dictionary with metrics derived from confusion matrix
    """
    # Convert one-hot encoded labels to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Convert probabilities to class indices
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get number of classes
    num_classes = cm.shape[0]
    
    # Initialize metrics
    metrics = {}
    
    # Calculate per-class metrics
    metrics['per_class'] = []
    
    for i in range(num_classes):
        # True positive for class i
        tp = cm[i, i]
        
        # False positives: sum of column i except true positive
        fp = np.sum(cm[:, i]) - tp
        
        # False negatives: sum of row i except true positive
        fn = np.sum(cm[i, :]) - tp
        
        # True negatives: sum of all elements except row i and column i
        tn = np.sum(cm) - tp - fp - fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics['per_class'].append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })
    
    # Calculate overall metrics
    metrics['overall'] = {
        'accuracy': np.sum(np.diag(cm)) / np.sum(cm),
        'macro_precision': np.mean([c['precision'] for c in metrics['per_class']]),
        'macro_recall': np.mean([c['recall'] for c in metrics['per_class']]),
        'macro_f1': np.mean([c['f1_score'] for c in metrics['per_class']]),
        'confusion_matrix': cm
    }
    
    return metrics

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        epsilon: Small constant for numerical stability
        
    Returns:
        Cross-entropy loss
    """
    # Ensure probabilities are in valid range
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # If y_true is not one-hot encoded, convert it
    if len(y_true.shape) == 1:
        num_classes = y_pred.shape[1]
        y_true_one_hot = np.zeros((y_true.size, num_classes))
        y_true_one_hot[np.arange(y_true.size), y_true] = 1
        y_true = y_true_one_hot
    
    # Calculate cross-entropy loss
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean squared error
    """
    return np.mean(np.square(y_pred - y_true))

def evaluate_model(model, X, y, batch_size=64, metrics_list=None):
    """
    Evaluate a model on test data with multiple metrics.
    
    Args:
        model: Neural network model
        X: Input data
        y: True labels
        batch_size: Batch size for evaluation
        metrics_list: List of metric functions to compute (optional)
        
    Returns:
        Dictionary of evaluation results
    """
    # Define default metrics if not provided
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision_recall_f1', 'confusion_matrix']
    
    # Initialize results
    results = {}
    
    # Make predictions in batches
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    all_preds = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        X_batch = X[start_idx:end_idx]
        batch_preds = model.forward(X_batch, training=False)
        all_preds.append(batch_preds)
    
    # Concatenate predictions
    y_pred = np.vstack(all_preds)
    
    # Compute loss
    if len(y.shape) > 1:  # One-hot encoded
        results['loss'] = cross_entropy_loss(y, y_pred)
    else:
        # Convert to one-hot for loss calculation
        num_classes = y_pred.shape[1]
        y_one_hot = np.zeros((y.size, num_classes))
        y_one_hot[np.arange(y.size), y] = 1
        results['loss'] = cross_entropy_loss(y_one_hot, y_pred)
    
    # Compute metrics
    for metric in metrics_list:
        if metric == 'accuracy':
            results['accuracy'] = accuracy(y, y_pred)
        elif metric == 'precision_recall_f1':
            precision, recall, f1 = precision_recall_f1(y, y_pred)
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
        elif metric == 'confusion_matrix':
            cm_metrics = confusion_matrix_metrics(y, y_pred)
            results['confusion_matrix'] = cm_metrics['overall']['confusion_matrix']
            results['per_class_metrics'] = cm_metrics['per_class']
    
    return results
