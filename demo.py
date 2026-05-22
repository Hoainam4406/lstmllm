#!/usr/bin/env python
"""
Quick Demo: Minimal ablation study run for testing
Runs only a few episodes for quick validation
"""

import torch
import gymnasium as gym
import numpy as np
from pathlib import Path

from config import *
from models import StudentInferenceActor, TeacherLSTMActor, TeacherLLMActor
from distillation import IndependentTrainer, DistillationTrainer
from evaluation import PolicyEvaluator, AblationStudyComparison, get_model_size
from utils import set_seed, get_device
from llm_interface import get_llm_interface
from PO_MuJoCo import POMuJoCoWrapper


def demo_v1_minimal():
    """Minimal V1 demo"""
    print("\n" + "="*60)
    print("DEMO: V1 LSTM Only (2 iterations)")
    print("="*60)
    
    # Setup
    set_seed(SEED)
    device = get_device(DEVICE)
    
    env = gym.make(ENV_NAME)
    env = POMuJoCoWrapper(env)
    
    # Create student
    student = StudentInferenceActor(
        obs_dim=POMDP_OBS_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
    )
    
    # Quick training
    trainer = IndependentTrainer(student, device=device)
    print("Training student for 2 iterations (minimal demo)...")
    
    trainer.train(
        env=env,
        num_iterations=2,
        episodes_per_iteration=2,
        batches_per_iteration=5,
    )
    
    # Quick eval
    print("\nEvaluating...")
    evaluator = PolicyEvaluator(student, device=device)
    eval_metrics = evaluator.evaluate(env, num_episodes=1)
    
    print(f"✓ V1 Demo Complete!")
    print(f"  Reward: {eval_metrics['mean_reward']:.2f}")
    print(f"  Success: {eval_metrics['success_rate']:.0%}")
    
    env.close()
    return eval_metrics


def demo_v2_minimal():
    """Minimal V2 demo"""
    print("\n" + "="*60)
    print("DEMO: V2 LSTM + Teacher (2 iterations)")
    print("="*60)
    
    set_seed(SEED)
    device = get_device(DEVICE)
    
    # Create TWO environments: Teacher needs full state, Student needs POMDP
    env_full = gym.make(ENV_NAME)  # Full state (17-dim) for Teacher
    env_pomdp = gym.make(ENV_NAME)
    env_pomdp = POMuJoCoWrapper(env_pomdp)  # POMDP (8-dim) for Student
    
    # Models
    student = StudentInferenceActor(
        obs_dim=POMDP_OBS_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
    )
    
    teacher = TeacherLSTMActor(
        full_state_dim=FULL_STATE_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
    )
    
    # Quick training - pass both environments
    trainer = DistillationTrainer(student, teacher, device=device, use_llm=False)
    print("Training for 2 iterations (minimal demo)...")
    
    trainer.train(
        env_student=env_pomdp,
        env_teacher=env_full,
        num_iterations=2,
        collection_episodes=2,
        batches_per_iteration=5,
    )
    
    # Eval
    print("\nEvaluating...")
    evaluator = PolicyEvaluator(student, device=device)
    eval_metrics = evaluator.evaluate(env_pomdp, num_episodes=1)
    
    print(f"✓ V2 Demo Complete!")
    print(f"  Reward: {eval_metrics['mean_reward']:.2f}")
    print(f"  Success: {eval_metrics['success_rate']:.0%}")
    
    env_full.close()
    env_pomdp.close()
    return eval_metrics


def main():
    """Run minimal demos"""
    print("\n" + "="*70)
    print(" "*15 + "LSTM + LLM Ablation Study - QUICK DEMO")
    print("="*70)
    print("\nThis demo runs minimal iterations to validate setup (~ 2-3 minutes)\n")
    
    try:
        # Demo V1
        metrics_v1 = demo_v1_minimal()
        
        # Demo V2
        metrics_v2 = demo_v2_minimal()
        
        # Summary
        print("\n" + "="*70)
        print("DEMO SUMMARY")
        print("="*70)
        print(f"\nV1 Reward:  {metrics_v1['mean_reward']:.2f}")
        print(f"V2 Reward:  {metrics_v2['mean_reward']:.2f}")
        print(f"Improvement: {metrics_v2['mean_reward'] - metrics_v1['mean_reward']:+.2f}")
        
        print("\n✓ Demo successful! You can now run the full pipeline:")
        print("\n  python ablation_pipeline.py --variants all\n")
        
    except KeyboardInterrupt:
        print("\n\n✗ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("\nTroubleshooting:")
        print("  1. Run setup_check.py to verify dependencies")
        print("  2. Check config.py for correct settings")
        print("  3. Ensure MuJoCo environment is working")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
