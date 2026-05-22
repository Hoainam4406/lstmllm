#!/usr/bin/env python3
"""
Kaggle Execution Script for LSTM+LLM Ablation Study
Runs V1 and V2 (stable variants) with mock LLM for V3
"""

import os
import sys

# Set environment for MuJoCo
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Run pipeline with V1 and V2 only (proven stable)
if __name__ == "__main__":
    from ablation_pipeline import main
    
    print("=" * 60)
    print("LSTM + LLM Ablation Study - Kaggle Edition")
    print("=" * 60)
    print("\nRunning V1 (LSTM baseline) + V2 (Teacher-Student)...")
    print("V3 (LLM guidance) will use mock embeddings\n")
    
    sys.argv = ['kaggle_run.py', '--variants', 'V1', 'V2', 'V3']
    main()
    
    print("\n✓ Pipeline complete!")
    print("Results saved to results/ablation_study/ABLATION_COMPARISON.md")
