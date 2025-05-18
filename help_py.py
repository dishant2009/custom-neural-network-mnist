"""
Helper module for displaying usage information about the project.

This module provides utility functions for displaying help information,
which is especially useful when running the project in a container.
"""

import argparse
import os
import sys

def display_help():
    """
    Display help information about the project and its scripts.
    """
    print("\n=============================================================")
    print("                MNIST Neural Network Project                  ")
    print("=============================================================\n")
    
    print("This project implements a custom neural network for MNIST digit")
    print("classification, with distributed hyperparameter tuning and model")
    print("interpretability features.\n")
    
    print("Available scripts:")
    print("-----------------\n")
    
    print("1. train.py - Train a neural network on MNIST")
    print("   Example: python train.py --architecture mlp --epochs 10\n")
    
    print("2. evaluate.py - Evaluate and interpret a trained model")
    print("   Example: python evaluate.py --model_path models/model.npy --interpretability gradcam\n")
    
    print("3. tune_hyperparams.py - Perform hyperparameter tuning")
    print("   Example: python tune_hyperparams.py --method optuna --num_trials 50\n")
    
    print("For detailed usage options, run any script with the --help flag:")
    print("   python train.py --help\n")
    
    print("=============================================================\n")

def get_script_help(script_name):
    """
    Display help for a specific script by parsing its arguments.
    
    Args:
        script_name: Name of the script (train.py, evaluate.py, or tune_hyperparams.py)
    """
    # Create a backup of sys.argv
    original_argv = sys.argv.copy()
    
    # Set sys.argv to get help for the specified script
    sys.argv = [script_name, '--help']
    
    try:
        # Import and run the specified script
        if script_name == 'train.py':
            import train
        elif script_name == 'evaluate.py':
            import evaluate
        elif script_name == 'tune_hyperparams.py':
            import tune_hyperparams
        else:
            print(f"Unknown script: {script_name}")
    except SystemExit:
        # ArgumentParser's --help flag will trigger a SystemExit, which is expected
        pass
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def get_all_script_help():
    """
    Display help for all main scripts in the project.
    """
    scripts = ['train.py', 'evaluate.py', 'tune_hyperparams.py']
    
    for script in scripts:
        print(f"\n{'=' * 50}")
        print(f"HELP: {script}")
        print(f"{'=' * 50}\n")
        get_script_help(script)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display help information')
    parser.add_argument('--script', type=str, choices=['train.py', 'evaluate.py', 'tune_hyperparams.py'],
                        help='Display help for a specific script')
    parser.add_argument('--all', action='store_true', help='Display help for all scripts')
    
    args = parser.parse_args()
    
    if args.script:
        get_script_help(args.script)
    elif args.all:
        get_all_script_help()
    else:
        display_help()
