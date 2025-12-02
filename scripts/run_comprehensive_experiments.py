#!/usr/bin/env python3
"""
Script to run comprehensive experiments for all three methods across all movement types.

This script runs:
1. M-CELS explainer (uses all 48 channels)
2. Adaptive-MO-Modality explainer (modality-level grouping)
3. Adaptive-MO-Sensor explainer (sensor-level grouping)

For each movement type: squat, extension, gait
With 150 samples and 1000 maximum iterations for comprehensive analysis.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and print its status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run all three XAI methods for all movement types."""
    print("🚀 Starting Comprehensive XAI Experiment Suite")
    print("This script will run M-CELS, Adaptive-MO-Modality, and Adaptive-MO-Sensor")
    print("for all movement types (squat, extension, gait) with 100 samples each")
    
    # Check if we're in the correct directory
    if not os.path.exists('main.py'):
        print("❌ Error: main.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Configuration for comprehensive experiments
    movement_types = ['squat', 'extension', 'gait']
    methods = [
        ('mcels', 'M-CELS', {}),
        ('adaptive_multi', 'Adaptive-MO-Modality', {'group_level': 'modality'}),
        ('adaptive_multi', 'Adaptive-MO-Sensor', {'group_level': 'sensor'})
    ]
    
    # Base arguments for all experiments
    base_args = [
        'C:/Users/20235732/AppData/Local/miniconda3/envs/sktime-dev/python.exe', 'main.py',
        '--run_mode', 'batch',
        '--num_samples', '150',
        '--max_itr', '1000',
        '--importance_method', 'shap',
        '--enable_seed',
        '--seed_value', '42'
    ]
    
    results = []
    total_experiments = len(movement_types) * len(methods)
    current_experiment = 0
    
    print(f"\n📊 Total experiments to run: {total_experiments}")
    print("=" * 80)
    
    # Run all combinations
    for movement in movement_types:
        print(f"\n🏃 Processing {movement.upper()} movement")
        print("-" * 50)
        
        for algo, method_name, extra_args in methods:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Running {method_name} for {movement}")
            
            # Build command arguments
            experiment_args = base_args + [
                '--algo', algo,
                '--movement_type', movement
            ]
            
            # Add method-specific arguments
            for key, value in extra_args.items():
                experiment_args.extend([f'--{key}', value])
            
            experiment_name = f"{method_name} - {movement.capitalize()}"
            success = run_command(experiment_args, experiment_name)
            results.append((experiment_name, success))
    
    # Summary
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*80)
    
    # Group results by movement type
    for movement in movement_types:
        print(f"\n{movement.upper()} Movement:")
        movement_results = [(name, success) for name, success in results if movement.capitalize() in name]
        
        for name, success in movement_results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            method_only = name.split(" - ")[0]
            print(f"  {method_only:20}: {status}")
    
    # Overall statistics
    successful_runs = sum(1 for _, success in results if success)
    failed_runs = total_experiments - successful_runs
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {failed_runs}")
    print(f"  Success rate: {(successful_runs/total_experiments)*100:.1f}%")
    
    if successful_runs == total_experiments:
        print("\n🎉 All experiments completed successfully!")
        print("You can now run the paper analysis script to generate comprehensive results.")
        print("\nNext step: python scripts/generate_paper_results.py")
    else:
        print(f"\n⚠️ {failed_runs} experiments failed. Check the error messages above.")
        print("You may want to re-run the failed experiments individually.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())