"""
Main Ablation Study Pipeline
Runs all three variants of LSTM + LLM experiments
"""

import torch
import gymnasium as gym
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys

from config import *
from models import StudentInferenceActor, TeacherLSTMActor, TeacherLLMActor
from training import PPOTrainer
from distillation import DistillationTrainer, IndependentTrainer
from evaluation import PolicyEvaluator, AblationStudyComparison, get_model_size
from utils import set_seed, get_device, setup_experiment_dirs, MetricsLogger
from llm_interface import get_llm_interface
from PO_MuJoCo import POMuJoCoWrapper


def run_variant_v1(env, device, results_dir):
    """
    Variant 1: DL Baseline (LSTM only, no Teacher)
    Student learns independently via policy gradient
    """
    print("\n" + "="*60)
    print("VARIANT 1: LSTM Only (Baseline RL)")
    print("="*60)
    
    # Create student model
    student = StudentInferenceActor(
        obs_dim=POMDP_OBS_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        feat_encoder_dim=STUDENT_FEAT_ENCODER_DIM,
    )
    
    # Initialize trainer
    trainer = IndependentTrainer(
        student_model=student,
        lr=PPO_LR,
        device=device,
    )
    
    # Create metrics logger
    metrics_logger = MetricsLogger("V1_LSTM_Only", results_dir)
    
    # Training loop
    print(f"\nTraining for {NUM_EPISODES} episodes...")
    training_stats = trainer.train(
        env=env,
        num_iterations=NUM_EPISODES // 5,
        episodes_per_iteration=5,
        batches_per_iteration=20,
    )
    
    # Save model
    checkpoint_path = Path(results_dir) / "V1_LSTM_Only" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(checkpoint_path))
    
    # Evaluation
    print("\nEvaluating trained policy...")
    evaluator = PolicyEvaluator(
        model=student,
        device=device,
        use_llm=False,
    )
    
    eval_metrics = evaluator.evaluate(
        env=env,
        num_episodes=5,
        success_threshold=SUCCESS_EPISODE_REWARD,
    )
    
    # Model info
    size_mb, num_params = get_model_size(student)
    model_info = {"size_mb": size_mb, "num_params": num_params}
    
    print(f"\n✓ V1 Complete")
    print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    
    return eval_metrics, training_stats, model_info


def run_variant_v2(env_full, env_pomdp, device, results_dir):
    """
    Variant 2: With Teacher (LSTM only, no LLM)
    Student learns to mimic Teacher via distillation
    """
    print("\n" + "="*60)
    print("VARIANT 2: LSTM + Teacher (No LLM)")
    print("="*60)
    
    # Create models
    student = StudentInferenceActor(
        obs_dim=POMDP_OBS_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        feat_encoder_dim=STUDENT_FEAT_ENCODER_DIM,
    )
    
    teacher = TeacherLSTMActor(
        full_state_dim=FULL_STATE_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        feat_encoder_dim=TEACHER_LSTM_FEAT_ENCODER_DIM,
    )
    
    # Phase 1: Train Teacher with PPO
    print("\nPhase 1: Training Teacher...")
    ppo_trainer = PPOTrainer(
        teacher_model=teacher,
        learning_rate=PPO_LR,
        gamma=PPO_GAMMA,
        lambda_gae=PPO_LAMBDA,
        clip_ratio=PPO_CLIP_RATIO,
        num_epochs=PPO_NUM_EPOCHS,
        entropy_coef=PPO_ENTROPY_COEF,
        device=device,
        use_llm=False,
    )
    
    ppo_stats = ppo_trainer.train(
        env=env_full,
        num_updates=5,
        steps_per_update=500,
    )
    
    # Save teacher
    teacher_checkpoint = Path(results_dir) / "V2_LSTM_Teacher" / "teacher.pt"
    teacher_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save(str(teacher_checkpoint))
    
    # Phase 2: Train Student via Distillation
    print("\nPhase 2: Training Student via Distillation...")
    distill_trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        student_lr=DISTILLATION_LR,
        kl_weight=DISTILLATION_KL_WEIGHT,
        mse_weight=DISTILLATION_MSE_WEIGHT,
        device=device,
        use_llm=False,
    )
    
    distill_stats = distill_trainer.train(
        env_teacher=env_full,
        env_student=env_pomdp,
        num_iterations=5,
        collection_episodes=4,
        batches_per_iteration=20,
    )
    
    # Save student
    student_checkpoint = Path(results_dir) / "V2_LSTM_Teacher" / "student.pt"
    student_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    distill_trainer.save(str(student_checkpoint))
    
    # Evaluation
    print("\nEvaluating trained policy...")
    evaluator = PolicyEvaluator(
        model=student,
        device=device,
        use_llm=False,
    )
    
    eval_metrics = evaluator.evaluate(
        env=env_pomdp,
        num_episodes=5,
        success_threshold=SUCCESS_EPISODE_REWARD,
    )
    
    # Model info
    size_mb, num_params = get_model_size(student)
    model_info = {"size_mb": size_mb, "num_params": num_params}
    
    print(f"\n✓ V2 Complete")
    print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    
    return eval_metrics, distill_stats, model_info


def run_variant_v3(env_full, env_pomdp, device, results_dir):
    """
    Variant 3: Full System (LSTM + LLM guidance)
    Student learns from Teacher that uses LLM guidance
    """
    print("\n" + "="*60)
    print("VARIANT 3: Full System (LSTM + LLM Guidance)")
    print("="*60)
    
    # Initialize LLM interface
    print("\nInitializing LLM Interface...")
    llm_interface = get_llm_interface(
        use_ollama=True,
        model_name=OLLAMA_MODEL,
        host=OLLAMA_HOST.split(":")[0],
        port=int(OLLAMA_HOST.split(":")[1]),
    )
    
    # Create models
    student = StudentInferenceActor(
        obs_dim=POMDP_OBS_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        feat_encoder_dim=STUDENT_FEAT_ENCODER_DIM,
    )
    
    teacher = TeacherLLMActor(
        full_state_dim=FULL_STATE_DIM,
        llm_latent_dim=LLM_LATENT_DIM,
        act_dim=ACTION_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        feat_encoder_dim=TEACHER_LLM_FEAT_ENCODER_DIM,
    )
    
    # Phase 1: Train Teacher with PPO (using LLM guidance)
    print("\nPhase 1: Training Teacher with LLM Guidance...")
    ppo_trainer = PPOTrainer(
        teacher_model=teacher,
        learning_rate=PPO_LR,
        gamma=PPO_GAMMA,
        lambda_gae=PPO_LAMBDA,
        clip_ratio=PPO_CLIP_RATIO,
        num_epochs=PPO_NUM_EPOCHS,
        entropy_coef=PPO_ENTROPY_COEF,
        device=device,
        use_llm=True,
        llm_interface=llm_interface,
    )
    
    ppo_stats = ppo_trainer.train(
        env=env_full,
        num_updates=5,
        steps_per_update=500,
    )
    
    # Save teacher
    teacher_checkpoint = Path(results_dir) / "V3_LSTM_LLM" / "teacher_llm.pt"
    teacher_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save(str(teacher_checkpoint))
    
    # Phase 2: Train Student via Distillation
    print("\nPhase 2: Training Student via Distillation from Teacher+LLM...")
    distill_trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        student_lr=DISTILLATION_LR,
        kl_weight=DISTILLATION_KL_WEIGHT,
        mse_weight=DISTILLATION_MSE_WEIGHT,
        device=device,
        use_llm=True,
        llm_interface=llm_interface,
    )
    
    distill_stats = distill_trainer.train(
        env_teacher=env_full,
        env_student=env_pomdp,
        num_iterations=5,
        collection_episodes=4,
        batches_per_iteration=20,
    )
    
    # Save student
    student_checkpoint = Path(results_dir) / "V3_LSTM_LLM" / "student.pt"
    student_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    distill_trainer.save(str(student_checkpoint))
    
    # Evaluation
    print("\nEvaluating trained policy...")
    evaluator = PolicyEvaluator(
        model=student,
        device=device,
        use_llm=True,
        llm_interface=llm_interface,
    )
    
    eval_metrics = evaluator.evaluate(
        env=env_pomdp,
        num_episodes=5,
        success_threshold=SUCCESS_EPISODE_REWARD,
    )
    
    # Model info
    size_mb, num_params = get_model_size(student)
    model_info = {"size_mb": size_mb, "num_params": num_params}
    
    print(f"\n✓ V3 Complete")
    print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    
    return eval_metrics, distill_stats, model_info


def main():
    """Main pipeline execution"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="LSTM + LLM Ablation Study Pipeline")
    parser.add_argument("--variants", nargs="+", choices=["V1", "V2", "V3", "all"],
                        default=["all"], help="Which variants to run")
    parser.add_argument("--results-dir", type=str, default="./results/ablation_study",
                        help="Results directory")
    parser.add_argument("--render", action="store_true", help="Render environment")
    args = parser.parse_args()
    
    # Check Ollama availability for V3 if needed
    variants_to_run = ["V1", "V2", "V3"] if "all" in args.variants else args.variants
    if "V3" in variants_to_run:
        print("\n" + "="*60)
        print("Checking Ollama availability for V3...")
        print("="*60)
        try:
            import subprocess
            result = subprocess.run(
                ["python", "check_ollama.py"],
                cwd=Path.cwd(),
                capture_output=False,
                timeout=60
            )
            if result.returncode != 0:
                print("\n⚠️  Ollama check script had issues")
                print("V3 will attempt to run with mock interface if Ollama unavailable\n")
        except Exception as e:
            print(f"\n⚠️  Could not run Ollama check: {e}")
            print("Proceeding anyway - V3 will use mock if needed\n")
    
    # Setup
    set_seed(SEED)
    device = get_device(DEVICE)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LSTM + LLM Ablation Study Pipeline")
    print("="*60)
    print(f"Device: {device}")
    print(f"Results Directory: {results_dir}")
    print(f"Number of Episodes: {NUM_EPISODES}")
    
    # Create environments
    print("\nInitializing Environments...")
    # Teacher needs full state (17-dim), Student needs POMDP (8-dim)
    env_full = gym.make(ENV_NAME)
    env_pomdp = gym.make(ENV_NAME)
    env_pomdp = POMuJoCoWrapper(env_pomdp)
    
    # Run variants
    comparison = AblationStudyComparison()
    
    variants_to_run = ["V1", "V2", "V3"] if "all" in args.variants else args.variants
    
    if "V1" in variants_to_run:
        eval_v1, train_v1, info_v1 = run_variant_v1(env_pomdp, device, results_dir)
        comparison.add_variant_result("V1_LSTM_Only", eval_v1, train_v1, info_v1)
    
    if "V2" in variants_to_run:
        eval_v2, train_v2, info_v2 = run_variant_v2(env_full, env_pomdp, device, results_dir)
        comparison.add_variant_result("V2_LSTM_Teacher", eval_v2, train_v2, info_v2)
    
    if "V3" in variants_to_run:
        eval_v3, train_v3, info_v3 = run_variant_v3(env_full, env_pomdp, device, results_dir)
        comparison.add_variant_result("V3_LSTM_LLM", eval_v3, train_v3, info_v3)
    
    # Generate comparison report
    print("\n" + "="*60)
    print("Generating Comparison Report...")
    print("="*60)
    
    report_path = results_dir / "ABLATION_COMPARISON.md"
    comparison.generate_markdown_report(str(report_path))
    comparison.generate_comparison_plots(str(results_dir / "plots"))
    
    # Print final summary
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Report: {report_path}")
    print(f"Plots: {results_dir}/plots/")
    
    # Close environments
    env_full.close()
    env_pomdp.close()


if __name__ == "__main__":
    main()
