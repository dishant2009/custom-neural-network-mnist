"""
Implementation of SHAP and LIME for neural network interpretability.

SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)
are methods for explaining the predictions of black-box machine learning models.
These techniques help understand which features contribute most to the model's predictions.

References:
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", 2017
- LIME: Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier", 2016
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

class SHAP:
    """
    Implementation of SHAP (SHapley Additive exPlanations) for model interpretability.
    
    SHAP values measure feature importance by calculating the contribution of each
    feature to a prediction. SHAP values are based on the game theoretic concept
    of Shapley values, which fairly distribute the contribution among features.
    """
    def __init__(self, model, background_data, num_features=None):
        """
        Initialize SHAP with a model and background data.
        
        Args:
            model: Neural network model
            background_data: Representative background data for baseline expectations
            num_features: Number of features (optional, inferred from background_data)
        """
        self.model = model
        self.background_data = background_data
        
        if num_features is None:
            if len(background_data.shape) > 1:
                self.num_features = background_data.shape[1]
            else:
                self.num_features = background_data.shape[0]
        else:
            self.num_features = num_features
    
    def explain(self, input_data, num_samples=100, target_class=None):
        """
        Compute SHAP values for input data.
        
        Args:
            input_data: Input data point(s) to explain
            num_samples: Number of samples to approximate SHAP values
            target_class: Target class for classification models (defaults to predicted class)
            
        Returns:
            shap_values: SHAP values for each feature
        """
        # Ensure input_data is 2D array
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Number of samples to use (limited by background data size)
        num_samples = min(num_samples, len(self.background_data))
        
        # Get predictions for input data
        predictions = self.model.forward(input_data, training=False)
        
        # If target class is not specified, use the predicted class
        if target_class is None and len(predictions.shape) > 1:
            target_class = np.argmax(predictions[0])
        
        # Initialize SHAP values
        shap_values = np.zeros((input_data.shape[0], self.num_features))
        
        # For each input data point
        for j, instance in enumerate(input_data):
            # Randomly sample from background data
            background_indices = np.random.choice(
                len(self.background_data), num_samples, replace=False
            )
            background_samples = self.background_data[background_indices]
            
            # Compute baseline prediction (expected value)
            background_preds = self.model.forward(background_samples, training=False)
            if target_class is not None and len(background_preds.shape) > 1:
                background_preds = background_preds[:, target_class]
            baseline = np.mean(background_preds)
            
            # For each feature, calculate its contribution
            for feature_idx in range(self.num_features):
                feature_effect = 0
                
                # Sample permutations to estimate feature importance
                for i in range(num_samples):
                    # Randomly select a background sample
                    bg_idx = np.random.choice(num_samples)
                    bg_sample = background_samples[bg_idx].copy()
                    
                    # Create two samples: one with the feature and one without
                    with_feature = bg_sample.copy()
                    with_feature[feature_idx] = instance[feature_idx]
                    
                    # Compute predictions
                    with_pred = self.model.forward(np.array([with_feature]), training=False)
                    if target_class is not None and len(with_pred.shape) > 1:
                        with_pred = with_pred[0, target_class]
                    else:
                        with_pred = with_pred[0]
                    
                    # Get predictions without the feature
                    without_pred = background_preds[bg_idx]
                    if target_class is not None and len(background_preds.shape) > 1:
                        without_pred = without_pred[target_class]
                    
                    # Compute the effect
                    feature_effect += (with_pred - without_pred) / num_samples
                
                # Store the feature effect
                shap_values[j, feature_idx] = feature_effect
            
            # Ensure SHAP values sum to difference from baseline
            instance_pred = predictions[j]
            if target_class is not None and len(instance_pred.shape) > 0:
                instance_pred = instance_pred[target_class]
            correction = (instance_pred - baseline) - np.sum(shap_values[j])
            shap_values[j] += correction / self.num_features
        
        return shap_values
    
    def plot_feature_importance(self, shap_values, feature_names=None, top_n=20, save_path=None):
        """
        Plot feature importance based on SHAP values.
        
        Args:
            shap_values: SHAP values from explain()
            feature_names: List of feature names (optional)
            top_n: Number of top features to show
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure with feature importance plot
        """
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(shap_values.shape[1])]
        
        # Compute mean absolute SHAP values across samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_abs_shap)
        
        # Select top N features
        if top_n is not None and top_n < len(sorted_idx):
            sorted_idx = sorted_idx[-top_n:]
        
        # Plot feature importance
        plt.figure(figsize=(10, max(6, len(sorted_idx) * 0.3)))
        plt.barh(
            y=range(len(sorted_idx)),
            width=mean_abs_shap[sorted_idx],
            color='dodgerblue'
        )
        plt.yticks(
            range(len(sorted_idx)),
            [feature_names[i] for i in sorted_idx]
        )
        plt.xlabel('mean(|SHAP value|)')
        plt.ylabel('Feature')
        plt.title('Feature Importance Based on SHAP Values')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_shap_values(self, shap_values, instance_idx=0, feature_names=None, save_path=None):
        """
        Plot SHAP values for a single instance.
        
        Args:
            shap_values: SHAP values from explain()
            instance_idx: Index of the instance to plot
            feature_names: List of feature names (optional)
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure with SHAP values plot
        """
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(shap_values.shape[1])]
        
        # Get SHAP values for the specified instance
        instance_shap = shap_values[instance_idx]
        
        # Sort features by SHAP value magnitude
        sorted_idx = np.argsort(np.abs(instance_shap))
        
        # Plot SHAP values
        plt.figure(figsize=(10, max(6, len(sorted_idx) * 0.3)))
        
        # Create a color map based on SHAP value sign
        colors = ['red' if val < 0 else 'blue' for val in instance_shap[sorted_idx]]
        
        plt.barh(
            y=range(len(sorted_idx)),
            width=instance_shap[sorted_idx],
            color=colors
        )
        plt.yticks(
            range(len(sorted_idx)),
            [feature_names[i] for i in sorted_idx]
        )
        plt.xlabel('SHAP value')
        plt.ylabel('Feature')
        plt.title(f'SHAP Values for Instance {instance_idx}')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"SHAP values plot saved to {save_path}")
        
        return plt.gcf()


class LIME:
    """
    Implementation of LIME (Local Interpretable Model-agnostic Explanations).
    
    LIME explains model predictions by approximating the model locally with a
    simple, interpretable model (typically a linear model). It perturbs the input
    and observes the changes in prediction to understand the model's behavior.
    """
    def __init__(self, model, num_samples=1000, kernel_width=0.25):
        """
        Initialize LIME.
        
        Args:
            model: Neural network model
            num_samples: Number of perturbed samples to generate
            kernel_width: Width of the exponential kernel (controls locality)
        """
        self.model = model
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def explain(self, input_data, feature_names=None, target_class=None, num_features=10):
        """
        Generate LIME explanation for input data.
        
        Args:
            input_data: Input data point to explain
            feature_names: List of feature names (optional)
            target_class: Target class for classification models (defaults to predicted class)
            num_features: Number of top features to include in the explanation
            
        Returns:
            Tuple of (coefficients, intercept, feature_names, model_prediction)
        """
        # Ensure input_data is 2D array with a single sample
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Get model prediction for the input
        prediction = self.model.forward(input_data, training=False)
        
        # If target class is not specified, use the predicted class
        if target_class is None and len(prediction.shape) > 1:
            target_class = np.argmax(prediction[0])
            original_prediction = prediction[0, target_class]
        else:
            original_prediction = prediction[0]
        
        # Extract input dimensions
        num_features_total = input_data.shape[1]
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(num_features_total)]
        
        # Generate perturbed samples around the input data
        np.random.seed(42)  # For reproducibility
        perturbed_samples = np.zeros((self.num_samples, num_features_total))
        
        for i in range(self.num_samples):
            # Generate perturbations from a normal distribution
            perturbed_samples[i] = input_data[0] + np.random.normal(0, 0.1, num_features_total)
        
        # Get model predictions for perturbed samples
        perturbed_predictions = self.model.forward(perturbed_samples, training=False)
        
        # Extract predictions for the target class
        if target_class is not None and len(perturbed_predictions.shape) > 1:
            perturbed_predictions = perturbed_predictions[:, target_class]
        
        # Calculate distances from the original sample
        distances = np.sqrt(np.sum((perturbed_samples - input_data) ** 2, axis=1))
        
        # Calculate weights using exponential kernel
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        # Fit a weighted linear model
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(perturbed_samples, perturbed_predictions, sample_weight=weights)
        
        # Get the coefficients
        coefficients = linear_model.coef_
        
        # Get the intercept
        intercept = linear_model.intercept_
        
        # Sort features by importance (absolute coefficient value)
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        
        # Keep only the top N features
        sorted_indices = sorted_indices[:num_features]
        
        # Return the most influential features
        top_coefficients = coefficients[sorted_indices]
        top_feature_names = [feature_names[i] for i in sorted_indices]
        
        return top_coefficients, intercept, top_feature_names, original_prediction
    
    def plot_explanation(self, coefficients, feature_names, prediction, save_path=None):
        """
        Plot LIME explanation.
        
        Args:
            coefficients: Feature coefficients from explain()
            feature_names: Feature names corresponding to coefficients
            prediction: Model's prediction for the explained instance
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure with LIME explanation plot
        """
        # Sort by coefficient magnitude
        sorted_indices = np.argsort(np.abs(coefficients))
        
        # Create colors based on coefficient sign
        colors = ['red' if c < 0 else 'blue' for c in coefficients[sorted_indices]]
        
        # Plot
        plt.figure(figsize=(10, max(6, len(sorted_indices) * 0.3)))
        plt.barh(
            y=range(len(sorted_indices)),
            width=coefficients[sorted_indices],
            color=colors
        )
        plt.yticks(
            range(len(sorted_indices)),
            [feature_names[i] for i in sorted_indices]
        )
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.title(f'LIME Explanation (Prediction: {prediction:.4f})')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"LIME explanation plot saved to {save_path}")
        
        return plt.gcf()
