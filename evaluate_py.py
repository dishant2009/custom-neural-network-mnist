"""
Script for evaluating and interpreting trained neural network models on MNIST.

This script loads a trained model and evaluates its performance on the test set,
generating various interpretability visualizations based on the model architecture.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from neural_network.network import NeuralNetwork
from neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from neural_network.activations import ReLU, Softmax
from neural_network.losses import CategoricalCrossEntropy

from utils.data_loader import load_data, one_hot_encode, get_statistical_embeddings
from utils.visualization import plot_confusion_matrix, plot_mnist_images
from utils.metrics import evaluate_model

from interpretability.gradcam import GradCAM
from interpretability.shap_lime import SHAP, LIME

def load_model(model_path, architecture='mlp'):
    """
    Load a model from a saved weights file.
    
    Args:
        model_path: Path to the saved weights file
        architecture: Model architecture ('mlp' or 'cnn')
        
    Returns:
        Loaded model
    """
    # Create network structure based on architecture
    network = NeuralNetwork()
    
    if architecture == 'mlp':
        # Recreate MLP structure
        network.add(Dense(784, 256))
        network.add(ReLU())
        network.add(BatchNormalization(256))
        network.add(Dropout(0.2))
        network.add(Dense(256, 128))
        network.add(ReLU())
        network.add(BatchNormalization(128))
        network.add(Dropout(0.2))
        network.add(Dense(128, 10))
        network.add(Softmax())
    else:  # CNN
        # Recreate CNN structure
        network.add(Conv2D(1, 32, kernel_size=3, padding=1))
        network.add(ReLU())
        network.add(MaxPool2D(pool_size=2, stride=2))
        network.add(Conv2D(32, 64, kernel_size=3, padding=1))
        network.add(ReLU())
        network.add(MaxPool2D(pool_size=2, stride=2))
        network.add(Flatten())
        network.add(Dense(7 * 7 * 64, 128))
        network.add(ReLU())
        network.add(Dropout(0.2))
        network.add(Dense(128, 10))
        network.add(Softmax())
    
    # Load weights
    network.load_weights(model_path)
    
    # Set loss function (needed for evaluation)
    network.set_loss(CategoricalCrossEntropy())
    
    return network

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate neural network on MNIST')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='Network architecture (mlp or cnn)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    
    # Interpretability arguments
    parser.add_argument('--interpretability', type=str, default=None, 
                       choices=['gradcam', 'shap', 'lime'], help='Interpretability method')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of samples to use for interpretability visualization')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='interpretability_output',
                       help='Directory for output visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.architecture)
    
    # Load MNIST data
    print("Loading MNIST data...")
    if args.architecture == 'mlp':
        # Flatten images for MLP
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(flatten=True)
    else:  # CNN
        # Keep image shape for CNN
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(flatten=False)
    
    # One-hot encode labels
    y_test_one_hot = one_hot_encode(y_test)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test_one_hot, batch_size=args.batch_size)
    
    # Print metrics
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['loss']:.4f}")
    
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    if 'confusion_matrix' in metrics:
        cm_plot = plot_confusion_matrix(y_test, model.forward(X_test, training=False))
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        cm_plot.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
    
    # Plot example predictions
    indices = np.random.choice(len(X_test), 10, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    predictions = model.forward(X_samples, training=False)
    
    pred_plot = plot_mnist_images(X_samples, y_samples, predictions)
    pred_path = os.path.join(args.output_dir, "predictions.png")
    pred_plot.savefig(pred_path)
    print(f"Prediction examples saved to {pred_path}")
    
    # Interpretability
    if args.interpretability is not None:
        print(f"Generating {args.interpretability.upper()} visualizations...")
        
        if args.interpretability == 'gradcam' and args.architecture == 'cnn':
            # Find the last convolutional layer
            conv_layer_idx = None
            for i, layer in enumerate(model.layers):
                if isinstance(layer, Conv2D):
                    conv_layer_idx = i
            
            if conv_layer_idx is None:
                print("No convolutional layer found in the model")
                return
            
            # Create GradCAM visualizer
            gradcam = GradCAM(model, target_layer_idx=conv_layer_idx)
            
            # Select random test samples
            indices = np.random.choice(len(X_test), args.num_samples, replace=False)
            
            # Generate visualizations
            for i, idx in enumerate(indices):
                X_sample = X_test[idx:idx+1]
                y_sample = y_test[idx]
                
                # Generate Grad-CAM visualization
                fig, _ = gradcam.visualize(
                    X_sample, 
                    save_path=os.path.join(args.output_dir, f"gradcam_sample_{i}_class_{y_sample}.png")
                )
                plt.close(fig)
            
            print(f"Grad-CAM visualizations saved to {args.output_dir}")
        
        elif args.interpretability == 'shap' and args.architecture == 'mlp':
            # Create SHAP explainer
            # Use a subset of training data as background
            background_indices = np.random.choice(len(X_train), 100, replace=False)
            background_data = X_train[background_indices]
            
            shap_explainer = SHAP(model, background_data)
            
            # Select random test samples
            indices = np.random.choice(len(X_test), args.num_samples, replace=False)
            samples = X_test[indices]
            
            # Generate global feature importance plot using statistical embeddings
            if args.num_samples >= 10:
                # Get statistical embeddings for a larger set
                stat_indices = np.random.choice(len(X_test), 50, replace=False)
                X_stat = X_test[stat_indices]
                
                # Convert to original image shape for feature extraction
                if len(X_stat.shape) == 2:  # Flattened
                    X_img = X_stat.reshape(-1, 28, 28)
                else:
                    X_img = X_stat.reshape(-1, X_stat.shape[1], X_stat.shape[2])
                
                # Generate statistical embeddings
                X_stat_embed = get_statistical_embeddings(X_img)
                
                # Create feature names
                feature_names = [
                    'Mean', 'Std'] + \
                    [f'Row {i} Mean' for i in range(28)] + \
                    [f'Row {i} Std' for i in range(28)] + \
                    [f'Col {i} Mean' for i in range(28)] + \
                    [f'Col {i} Std' for i in range(28)] + \
                    [f'Q{q} {s}' for q in range(1, 5) for s in ['Mean', 'Std', 'Max', 'Min']]
                
                # Compute SHAP values for statistical features
                print("Computing SHAP values for statistical features...")
                shap_values = shap_explainer.explain(X_stat_embed[:10])
                
                # Plot feature importance
                shap_explainer.plot_feature_importance(
                    shap_values, 
                    feature_names=feature_names, 
                    save_path=os.path.join(args.output_dir, "shap_feature_importance.png")
                )
            
            # Generate individual SHAP explanations
            for i, idx in enumerate(indices):
                X_sample = X_test[idx:idx+1]
                y_sample = y_test[idx]
                
                # Compute SHAP values
                shap_values = shap_explainer.explain(X_sample)
                
                # Create pixel feature names
                if args.architecture == 'mlp':
                    feature_names = [f'Pixel {i}' for i in range(X_sample.shape[1])]
                
                # Plot SHAP values
                shap_explainer.plot_shap_values(
                    shap_values,
                    feature_names=feature_names,
                    save_path=os.path.join(args.output_dir, f"shap_sample_{i}_class_{y_sample}.png")
                )
            
            print(f"SHAP visualizations saved to {args.output_dir}")
        
        elif args.interpretability == 'lime' and args.architecture == 'mlp':
            # Create LIME explainer
            lime_explainer = LIME(model)
            
            # Select random test samples
            indices = np.random.choice(len(X_test), args.num_samples, replace=False)
            
            # Generate visualizations
            for i, idx in enumerate(indices):
                X_sample = X_test[idx:idx+1]
                y_sample = y_test[idx]
                
                # Create pixel feature names
                if args.architecture == 'mlp':
                    feature_names = [f'Pixel {i}' for i in range(X_sample.shape[1])]
                
                # Compute LIME explanation
                coefficients, intercept, feature_names, prediction = lime_explainer.explain(
                    X_sample, feature_names, num_features=30
                )
                
                # Plot explanation
                lime_explainer.plot_explanation(
                    coefficients, 
                    feature_names, 
                    prediction,
                    save_path=os.path.join(args.output_dir, f"lime_sample_{i}_class_{y_sample}.png")
                )
            
            print(f"LIME visualizations saved to {args.output_dir}")
        
        else:
            print(f"Interpretability method '{args.interpretability}' not applicable for '{args.architecture}' architecture")
    
    print("Evaluation completed successfully!")

if __name__ == '__main__':
    main()
