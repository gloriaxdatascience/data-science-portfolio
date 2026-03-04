#!/usr/bin/env python
"""
Viral Cooking Trends Pipeline
Run: python run_pipeline.py
"""

import subprocess
import sys
import os
from datetime import datetime

def print_step(message):
    print(f"\n{'='*60}")
    print(f"📌 {message}")
    print(f"{'='*60}\n")

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    print(f"✅ Success")
    return result

def main():
    print_step("VIRAL COOKING TRENDS PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Step 1: Data Collection
    print_step("STEP 1: COLLECTING YOUTUBE DATA")
    run_command("python src/youtube_api.py")
    
    # Step 2: Build Time Series
    print_step("STEP 2: BUILDING TIME SERIES")
    run_command("python src/time_series.py")
    
    # Step 3: Feature Engineering
    print_step("STEP 3: ENGINEERING FEATURES")
    run_command("python src/features.py")
    
    # Step 4: Generate Visualizations
    print_step("STEP 4: GENERATING VISUALIZATIONS")
    run_command("python src/visualize.py")
    
    # Step 5: Run Jupyter Notebook (optional)
    print_step("STEP 5: RUNNING FINAL ANALYSIS")
    run_command("jupyter nbconvert --execute --to notebook --inplace notebooks/03_final_analysis.ipynb")
    
    print_step("PIPELINE COMPLETE! 🎉")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nCheck these outputs:")
    print("  📊 Figures: data/figures/")
    print("  📈 Processed data: data/processed/")
    print("  📝 Report: reports/final_analysis.md")

if __name__ == "__main__":
    main()