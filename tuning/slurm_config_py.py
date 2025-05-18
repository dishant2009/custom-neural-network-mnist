"""
Utilities for running hyperparameter tuning on SLURM clusters.

SLURM (Simple Linux Utility for Resource Management) is a workload manager
used in high-performance computing environments. This module provides utilities
for submitting and managing hyperparameter tuning jobs on SLURM clusters.
"""

import os
import subprocess
import argparse
import numpy as np
import json

def generate_slurm_script(job_name, output_dir, python_script, num_trials=10, 
                          time='12:00:00', partition='gpu', gpus_per_task=1, 
                          cpus_per_task=4, mem_per_task='16G'):
    """
    Generate a SLURM submission script.
    
    Args:
        job_name: Name of the SLURM job
        output_dir: Directory for job output files
        python_script: Python script to run
        num_trials: Number of hyperparameter trials to run
        time: Maximum job runtime in format 'HH:MM:SS'
        partition: SLURM partition/queue to use
        gpus_per_task: Number of GPUs to allocate per task
        cpus_per_task: Number of CPU cores to allocate per task
        mem_per_task: Memory to allocate per task (e.g., '16G')
        
    Returns:
        Path to the generated SLURM script
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create script path
    script_path = os.path.join(output_dir, f"{job_name}_slurm.sh")
    
    # SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}_%A_%a.out
#SBATCH --error={output_dir}/{job_name}_%A_%a.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus_per_task}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_per_task}
#SBATCH --array=0-{num_trials-1}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Started at: $(date)"

# Load modules (customize based on your HPC environment)
module load python/3.8
module load cuda/11.2

# Activate virtual environment (customize based on your environment)
source ~/venv/bin/activate

# Run the script with trial ID as an argument
python {python_script} --trial_id $SLURM_ARRAY_TASK_ID

echo "Finished at: $(date)"
"""
    
    # Write script to file
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Generated SLURM script: {script_path}")
    return script_path

def submit_slurm_job(script_path):
    """
    Submit a SLURM job using the generated script.
    
    Args:
        script_path: Path to the SLURM submission script
        
    Returns:
        Job ID if submission was successful, None otherwise
    """
    try:
        # Submit job using sbatch
        result = subprocess.run(['sbatch', script_path], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        # Extract job ID from output
        # Expected output format: "Submitted batch job 123456"
        job_id = result.stdout.strip().split()[-1]
        print(f"Successfully submitted SLURM job with ID: {job_id}")
        return job_id
    
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}")
        print(f"Error output: {e.stderr}")
        return None

def collect_results(output_dir, job_name, num_trials):
    """
    Collect and aggregate results from completed SLURM jobs.
    
    Args:
        output_dir: Directory containing job output files
        job_name: Name of the SLURM job
        num_trials: Number of hyperparameter trials that were run
        
    Returns:
        Dictionary with aggregated results
    """
    results = []
    
    for trial_id in range(num_trials):
        # Look for results file for this trial
        result_file = os.path.join(output_dir, f"result_{job_name}_{trial_id}.json")
        
        if os.path.exists(result_file):
            # Load results from JSON file
            with open(result_file, 'r') as f:
                trial_result = json.load(f)
            
            results.append(trial_result)
    
    if not results:
        print("No results found. Jobs may still be running or failed.")
        return None
    
    # Find best trial
    if 'val_acc' in results[0]:
        # For accuracy, higher is better
        best_idx = np.argmax([r['val_acc'] for r in results])
        metric = 'val_acc'
    elif 'val_loss' in results[0]:
        # For loss, lower is better
        best_idx = np.argmin([r['val_loss'] for r in results])
        metric = 'val_loss'
    else:
        # If no known metric found, use the first result
        best_idx = 0
        metric = 'unknown'
    
    best_trial = results[best_idx]
    
    # Aggregate results
    aggregated = {
        'num_completed_trials': len(results),
        'best_trial': best_trial,
        'best_metric': f"{metric}: {best_trial.get(metric, 'N/A')}",
        'all_results': results
    }
    
    return aggregated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and submit SLURM job for hyperparameter tuning')
    parser.add_argument('--job_name', type=str, default='mnist_tune', help='Name of the SLURM job')
    parser.add_argument('--output_dir', type=str, default='logs', help='Directory for job output files')
    parser.add_argument('--script', type=str, default='tune_hyperparams.py', help='Python script to run')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of hyperparameter trials to run')
    parser.add_argument('--submit', action='store_true', help='Submit the job after generating the script')
    args = parser.parse_args()
    
    # Generate SLURM script
    script_path = generate_slurm_script(
        job_name=args.job_name,
        output_dir=args.output_dir,
        python_script=args.script,
        num_trials=args.num_trials
    )
    
    # Submit job if requested
    if args.submit:
        job_id = submit_slurm_job(script_path)
        if job_id:
            print(f"Monitor your job with: squeue -j {job_id}")
