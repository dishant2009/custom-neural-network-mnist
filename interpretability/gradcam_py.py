"""
Implementation of Grad-CAM for CNN model interpretability.

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for visualizing
which regions of an input image are important for a specific prediction made by a CNN.
It uses the gradients flowing into the final convolutional layer to produce a coarse
localization map highlighting important regions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", 2017.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network.layers import Conv2D

class GradCAM:
    """
    Implementation of Grad-CAM for visualizing CNN decisions.
    
    Grad-CAM uses the gradients of any target concept (e.g., "dog" class score) flowing
    into the final convolutional layer to produce a coarse localization map highlighting
    important regions in the image for predicting the concept.
    """
    def __init__(self, model, target_layer_idx):
        """
        Initialize GradCAM with a model and target layer.
        
        Args:
            model: Neural network model
            target_layer_idx: Index of the target convolutional layer (usually the last conv layer)
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        
        # Get the target layer
        self.target_layer = model.layers[target_layer_idx]
        
        # Ensure the target layer is a convolutional layer
        if not isinstance(self.target_layer, Conv2D):
            raise ValueError("Target layer must be a convolutional layer (Conv2D)")
            
        self.gradients = None
        self.activations = None
    
    def compute_gradients(self, input_image, target_class=None):
        """
        Compute gradients of the target class with respect to the target layer activations.
        
        Args:
            input_image: Input image of shape (1, channels, height, width)
            target_class: Target class index. If None, use the predicted class.
            
        Returns:
            The predicted or target class
        """
        # Store the original input image
        self.input_image = input_image
        
        # Forward pass to get activations and predictions
        self.activations = None
        
        # Forward pass up to the target layer to get activations
        x = input_image
        for i, layer in enumerate(self.model.layers):
            if i == self.target_layer_idx:
                if isinstance(layer, Conv2D):
                    self.activations = layer.forward(x)
                    break
            
            # Forward through the layer
            if hasattr(layer, 'training'):
                x = layer.forward(x, training=False)
            else:
                x = layer.forward(x)
        
        # Continue forward pass after target layer to get predictions
        for i in range(self.target_layer_idx + 1, len(self.model.layers)):
            layer = self.model.layers[i]
            if hasattr(layer, 'training'):
                x = layer.forward(x, training=False)
            else:
                x = layer.forward(x)
        
        # Final predictions
        predictions = x
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = np.argmax(predictions[0])
        
        # One-hot encode the target class
        one_hot = np.zeros_like(predictions)
        one_hot[0, target_class] = 1
        
        # Initialize gradients for backpropagation
        self.gradients = np.zeros_like(self.activations)
        
        # Backward pass from the output (one-hot encoded target class)
        dout = one_hot
        
        # Backward pass through layers after target layer
        for i in range(len(self.model.layers) - 1, self.target_layer_idx, -1):
            dout = self.model.layers[i].backward(dout)
        
        # Store gradients at the target layer
        self.gradients = dout
        
        return target_class
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate a class activation map for the input image.
        
        Args:
            input_image: Input image of shape (1, channels, height, width)
            target_class: Target class index. If None, use the predicted class.
            
        Returns:
            cam: Class activation map of shape (height, width)
            target_class: The predicted or provided target class
        """
        # Compute gradients if not already computed
        target_class = self.compute_gradients(input_image, target_class)
        
        # Calculate channel-wise weights by global average pooling the gradients
        # Shape: (output_channels,)
        weights = np.mean(self.gradients, axis=(0, 2, 3))
        
        # Create a weighted combination of feature maps
        # Initialize CAM with zeros (height, width)
        cam = np.zeros(self.activations.shape[2:4], dtype=np.float32)
        
        # Weighted sum of activation maps
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
        
        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize CAM for visualization
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # Add small constant to avoid division by zero
        
        return cam, target_class
    
    def resize_cam(self, cam, target_size):
        """
        Resize the class activation map to match target size.
        
        Args:
            cam: Class activation map
            target_size: Tuple of (height, width) for resizing
            
        Returns:
            Resized class activation map
        """
        # Simple nearest-neighbor upsampling
        cam_h, cam_w = cam.shape
        target_h, target_w = target_size
        
        # Scale factors
        h_scale = target_h / cam_h
        w_scale = target_w / cam_w
        
        # Initialize resized CAM
        resized_cam = np.zeros((target_h, target_w))
        
        # Resize using nearest-neighbor interpolation
        for i in range(target_h):
            for j in range(target_w):
                # Find corresponding pixel in original CAM
                orig_i = min(int(i / h_scale), cam_h - 1)
                orig_j = min(int(j / w_scale), cam_w - 1)
                resized_cam[i, j] = cam[orig_i, orig_j]
        
        return resized_cam
    
    def overlay_cam(self, input_image, cam, alpha=0.5):
        """
        Overlay the class activation map on the input image.
        
        Args:
            input_image: Input image (height, width, channels) or (height, width)
            cam: Class activation map
            alpha: Transparency factor for overlay
            
        Returns:
            overlay: Overlayed image
        """
        # Reshape and normalize input image if needed
        if len(input_image.shape) == 3:
            # (channels, height, width) -> (height, width, channels)
            if input_image.shape[0] == 1 or input_image.shape[0] == 3:
                input_image = np.transpose(input_image, (1, 2, 0))
        
        # Ensure input image is normalized to [0, 1]
        if input_image.max() > 1:
            input_image = input_image / 255.0
        
        # Resize CAM to match input image dimensions
        if cam.shape != input_image.shape[:2]:
            cam = self.resize_cam(cam, input_image.shape[:2])
        
        # Create a heatmap using the 'jet' colormap
        # This converts CAM values to RGB colors
        heatmap = plt.cm.jet(cam)[:, :, :3]
        
        # Convert grayscale image to RGB if needed
        if len(input_image.shape) == 2:
            input_image = np.repeat(input_image[:, :, np.newaxis], 3, axis=2)
        elif input_image.shape[2] == 1:
            input_image = np.repeat(input_image, 3, axis=2)
        
        # Create the overlay
        overlay = (1 - alpha) * input_image + alpha * heatmap
        
        # Ensure overlay is in range [0, 1]
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def visualize(self, input_image, target_class=None, save_path=None):
        """
        Generate and visualize Grad-CAM for the input image.
        
        Args:
            input_image: Input image
            target_class: Target class index
            save_path: Path to save the visualization (optional)
            
        Returns:
            fig: Matplotlib figure with the visualization
        """
        # Generate CAM
        cam, predicted_class = self.generate_cam(input_image, target_class)
        
        # Get original image (first channel if multi-channel)
        if input_image.shape[1] == 1:  # Grayscale
            orig_img = input_image[0, 0]
        else:  # Color (take first 3 channels)
            orig_img = np.transpose(input_image[0, :3], (1, 2, 0))
            
            # Normalize if needed
            if orig_img.max() > 1:
                orig_img = orig_img / 255.0
        
        # Create overlay
        overlay = self.overlay_cam(orig_img, cam)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
            axes[0].imshow(orig_img)
        else:
            axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title(f'Original Image\nPredicted Class: {predicted_class}')
        axes[0].axis('off')
        
        # CAM
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        return fig, cam
