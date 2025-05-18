"""
MLflow experiment tracking for neural networks.

MLflow is an open-source platform for managing the ML lifecycle, including experimentation,
reproducibility, and deployment. This module provides utilities for tracking
neural network experiments using MLflow.
"""

import mlflow
import os
import time
import matplotlib.pyplot as plt
import numpy as np

class MLflowTracker:
    """
    Class for tracking neural network experiments using MLflow.
    
    This class provides a convenient interface for logging parameters, metrics,
    and artifacts during the training and evaluation of neural networks.
    """
    def __init__(self, experiment_name, tracking_uri="file:./mlruns"):
        """
        Initialize MLflow experiment tracking.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server (default: local file system)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        print(f"MLflow experiment '{experiment_name}' initialized with ID: {self.experiment_id}")
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run (default: auto-generated based on timestamp)
        """
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        print(f"Started MLflow run: {run_name}")
    
    def log_params(self, params):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step/epoch number (optional)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, name):
        """
        Log a model to MLflow.
        
        Args:
            model: Neural network model to log
            name: Name for the model
        """
        # Create directory for model if it doesn't exist
        model_dir = f"models/{name}"
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        
        # Save model weights
        model_path = model_dir + ".npy"
        model.save_weights(model_path)
        
        # Log model as an artifact
        mlflow.log_artifact(model_path)
        print(f"Logged model: {name}")
    
    def log_artifact(self, artifact_path):
        """
        Log an artifact to MLflow.
        
        Args:
            artifact_path: Path to the artifact file
        """
        mlflow.log_artifact(artifact_path)
    
    def log_figure(self, figure, filename):
        """
        Log a matplotlib figure to MLflow.
        
        Args:
            figure: Matplotlib figure to log
            filename: Name for the figure file
        """
        # Create directory for figures if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        
        # Save figure
        figure_path = f"figures/{filename}"
        figure.savefig(figure_path)
        plt.close(figure)
        
        # Log figure as an artifact
        mlflow.log_artifact(figure_path)
    
    def log_history(self, history):
        """
        Log training history and generate plots.
        
        Args:
            history: Dictionary containing training history (loss and accuracy)
        """
        # Log metrics
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
            history['train_loss'], history['train_acc'], 
            history['val_loss'], history['val_acc']
        )):
            self.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
        
        # Create and log loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        self.log_figure(plt.gcf(), 'loss_plot.png')
        
        # Create and log accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        self.log_figure(plt.gcf(), 'accuracy_plot.png')
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print("MLflow run ended")
    
    def get_best_run(self, metric_name, mode='max'):
        """
        Get the best run based on a specific metric.
        
        Args:
            metric_name: Name of the metric to optimize
            mode: 'max' for metrics where higher is better, 'min' for metrics where lower is better
            
        Returns:
            Information about the best run
        """
        # Search for runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if runs.empty:
            print("No runs found for this experiment")
            return None
        
        # Find the best run based on the metric
        metric_key = f'metrics.{metric_name}'
        if metric_key not in runs.columns:
            print(f"Metric '{metric_name}' not found in any run")
            return None
        
        if mode == 'max':
            # Find run with maximum metric value
            best_run_idx = runs[metric_key].idxmax()
        else:
            # Find run with minimum metric value
            best_run_idx = runs[metric_key].idxmin()
        
        best_run = runs.loc[best_run_idx]
        
        print(f"Best run for '{metric_name}' ({mode}): {best_run['run_id']}")
        print(f"  {metric_name}: {best_run[metric_key]}")
        
        return best_run
    
    def load_model_from_run(self, run_id, model_name):
        """
        Load a model from a specific run.
        
        Args:
            run_id: ID of the run
            model_name: Name of the model
            
        Returns:
            Path to the downloaded model file
        """
        # Download the model artifact
        client = mlflow.tracking.MlflowClient()
        artifact_path = f"{model_name}.npy"
        
        # Create local directory for the model
        local_dir = f"downloaded_models/{run_id}"
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the artifact
        local_path = client.download_artifacts(run_id, artifact_path, local_dir)
        
        print(f"Downloaded model from run {run_id} to {local_path}")
        return local_path
