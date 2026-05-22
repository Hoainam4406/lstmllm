"""
Evaluation and Metrics for Ablation Study
Collects metrics and generates comparison reports
"""

import torch
import numpy as np
import gymnasium as gym
import time
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from models import StudentInferenceActor
from llm_interface import OllamaInterface
from config import *


class PolicyEvaluator:
    """Evaluate policy performance on environment"""
    
    def __init__(
        self,
        model: StudentInferenceActor,
        device: str = "cuda",
        use_llm: bool = False,
        llm_interface: OllamaInterface = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_llm = use_llm
        self.llm_interface = llm_interface
        self.model.eval()
    
    def evaluate(
        self,
        env: gym.Env,
        num_episodes: int = 5,
        render: bool = False,
        success_threshold: float = SUCCESS_EPISODE_REWARD,
    ) -> Dict[str, Any]:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            env: Environment
            num_episodes: Number of evaluation episodes
            render: Whether to render environment
            success_threshold: Reward threshold for success
        
        Returns:
            Evaluation metrics
        """
        
        episode_rewards = []
        episode_lengths = []
        inference_times = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            hidden_state = None
            
            episode_reward = 0.0
            episode_length = 0
            
            done = False
            episode_times = []
            
            while not done:
                if render:
                    env.render()
                
                # Get partial observation
                pomdp_obs = obs[:POMDP_OBS_DIM]
                
                # Inference with timing
                start_time = time.time()
                
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(pomdp_obs).float().to(self.device)
                    obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                    
                    if self.use_llm and self.llm_interface:
                        llm_z = self.llm_interface.generate_strategy_embedding(obs)
                        llm_z = llm_z.unsqueeze(0).unsqueeze(0)
                        # Note: This is simplified; actual inference depends on model architecture
                    
                    dist, _, hidden_state = self.model(obs_tensor, hidden_state)
                    action = dist.sample()
                
                inference_time = (time.time() - start_time) * 1000  # ms
                episode_times.append(inference_time)
                
                # Environment step
                action_np = action.squeeze().cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            inference_times.extend(episode_times)
            
            if episode_reward >= success_threshold:
                success_count += 1
        
        return {
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "success_rate": float(success_count / num_episodes),
            "mean_inference_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
            "std_inference_time_ms": float(np.std(inference_times)) if inference_times else 0.0,
            "episode_rewards": episode_rewards,
        }


class AblationStudyComparison:
    """Compare results across ablation variants"""
    
    def __init__(self):
        self.results = {}
        self.training_logs = {}
    
    def add_variant_result(
        self,
        variant_name: str,
        eval_metrics: Dict[str, Any],
        training_metrics: Dict[str, Any] = None,
        model_info: Dict[str, Any] = None,
    ):
        """
        Add evaluation results for a variant.
        
        Args:
            variant_name: Name of variant (e.g., "V1_LSTM_Only")
            eval_metrics: Evaluation metrics dict
            training_metrics: Training history dict
            model_info: Model info (size, params, etc.)
        """
        self.results[variant_name] = {
            "eval": eval_metrics,
            "training": training_metrics or {},
            "model_info": model_info or {},
        }
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table across variants.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for variant_name, result in self.results.items():
            eval_metrics = result["eval"]
            
            row = {
                "Variant": variant_name,
                "Mean Reward": f"{eval_metrics['mean_reward']:.2f}",
                "Std Reward": f"{eval_metrics['std_reward']:.2f}",
                "Success Rate": f"{eval_metrics['success_rate']:.1%}",
                "Mean Length": f"{eval_metrics['mean_episode_length']:.0f}",
                "Avg Inference (ms)": f"{eval_metrics['mean_inference_time_ms']:.2f}",
            }
            
            if result["model_info"]:
                row["Model Size (MB)"] = f"{result['model_info'].get('size_mb', 0):.2f}"
                row["Num Params"] = f"{result['model_info'].get('num_params', 0):,}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to markdown table without tabulate dependency.
        
        Args:
            df: DataFrame to convert
        
        Returns:
            Markdown formatted table
        """
        # Create header row
        markdown = "| " + " | ".join(df.columns.astype(str)) + " |\n"
        markdown += "|" + "|".join(["---"] * len(df.columns)) + "|\n"
        
        # Create data rows
        for _, row in df.iterrows():
            markdown += "| " + " | ".join(row.astype(str)) + " |\n"
        
        return markdown
    
    def generate_markdown_report(self, output_path: str = "comparison_report.md"):
        """
        Generate markdown comparison report.
        
        Args:
            output_path: Output file path
        """
        report = "# LSTM + LLM Ablation Study Comparison\n\n"
        
        # Summary table
        report += "## Evaluation Results\n\n"
        table = self.get_comparison_table()
        
        # Convert table to markdown manually (avoid tabulate dependency)
        report += self._dataframe_to_markdown(table) + "\n\n"
        
        # Detailed metrics for each variant
        report += "## Detailed Results\n\n"
        
        for variant_name, result in self.results.items():
            eval_metrics = result["eval"]
            
            report += f"### {variant_name}\n\n"
            report += "**Evaluation Metrics:**\n"
            report += f"- Mean Episode Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}\n"
            report += f"- Max Reward: {eval_metrics['max_reward']:.2f}\n"
            report += f"- Min Reward: {eval_metrics['min_reward']:.2f}\n"
            report += f"- Success Rate (>{SUCCESS_EPISODE_REWARD}): {eval_metrics['success_rate']:.1%}\n"
            report += f"- Mean Episode Length: {eval_metrics['mean_episode_length']:.0f}\n"
            report += f"- Mean Inference Time: {eval_metrics['mean_inference_time_ms']:.2f} ms\n"
            report += f"- Std Inference Time: {eval_metrics['std_inference_time_ms']:.2f} ms\n"
            
            if result["model_info"]:
                report += "\n**Model Info:**\n"
                report += f"- Model Size: {result['model_info'].get('size_mb', 0):.2f} MB\n"
                report += f"- Number of Parameters: {result['model_info'].get('num_params', 0):,}\n"
            
            report += "\n"
        
        # Analysis and conclusions
        report += "## Analysis\n\n"
        
        best_reward_variant = max(
            self.results.items(),
            key=lambda x: x[1]["eval"]["mean_reward"]
        )
        report += f"**Best Mean Reward:** {best_reward_variant[0]} "
        report += f"({best_reward_variant[1]['eval']['mean_reward']:.2f})\n\n"
        
        best_success_variant = max(
            self.results.items(),
            key=lambda x: x[1]["eval"]["success_rate"]
        )
        report += f"**Best Success Rate:** {best_success_variant[0]} "
        report += f"({best_success_variant[1]['eval']['success_rate']:.1%})\n\n"
        
        best_speed_variant = min(
            self.results.items(),
            key=lambda x: x[1]["eval"]["mean_inference_time_ms"]
        )
        report += f"**Fastest Inference:** {best_speed_variant[0]} "
        report += f"({best_speed_variant[1]['eval']['mean_inference_time_ms']:.2f} ms)\n\n"
        
        # Conclusion
        report += "## Conclusion\n\n"
        report += "### Key Findings:\n"
        report += "1. **Impact of Teacher (V2 vs V1):** "
        if self.results.get("V2_LSTM_Teacher"):
            v2_reward = self.results["V2_LSTM_Teacher"]["eval"]["mean_reward"]
            v1_reward = self.results.get("V1_LSTM_Only", {}).get("eval", {}).get("mean_reward", 0)
            if v1_reward > 0:
                improvement = ((v2_reward - v1_reward) / abs(v1_reward)) * 100
                report += f"V2 achieves {improvement:+.1f}% change in reward.\n"
        report += "\n"
        
        report += "2. **Impact of LLM (V3 vs V2):** "
        if self.results.get("V3_LSTM_LLM") and self.results.get("V2_LSTM_Teacher"):
            v3_reward = self.results["V3_LSTM_LLM"]["eval"]["mean_reward"]
            v2_reward = self.results["V2_LSTM_Teacher"]["eval"]["mean_reward"]
            if v2_reward > 0:
                improvement = ((v3_reward - v2_reward) / abs(v2_reward)) * 100
                report += f"V3 achieves {improvement:+.1f}% change in reward.\n"
        report += "\n"
        
        report += "3. **Computational Cost:** "
        if self.results:
            slowest = max(
                self.results.items(),
                key=lambda x: x[1]["eval"]["mean_inference_time_ms"]
            )
            fastest = min(
                self.results.items(),
                key=lambda x: x[1]["eval"]["mean_inference_time_ms"]
            )
            slowdown = slowest[1]["eval"]["mean_inference_time_ms"] / (fastest[1]["eval"]["mean_inference_time_ms"] + 1e-6)
            report += f"{slowest[0]} is {slowdown:.1f}x slower than {fastest[0]}.\n"
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")
        return report
    
    def generate_comparison_plots(self, output_dir: str = "./results/plots"):
        """
        Generate comparison plots.
        
        Args:
            output_dir: Output directory for plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        variant_names = list(self.results.keys())
        mean_rewards = [self.results[v]["eval"]["mean_reward"] for v in variant_names]
        std_rewards = [self.results[v]["eval"]["std_reward"] for v in variant_names]
        success_rates = [self.results[v]["eval"]["success_rate"] * 100 for v in variant_names]
        inference_times = [self.results[v]["eval"]["mean_inference_time_ms"] for v in variant_names]
        
        # Plot 1: Reward Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(variant_names))
        ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel("Variant")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Mean Reward Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variant_names, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reward_comparison.png", dpi=150)
        plt.close()
        
        # Plot 2: Success Rate Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_pos, success_rates, alpha=0.7, color='green')
        ax.set_xlabel("Variant")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variant_names, rotation=45)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_comparison.png", dpi=150)
        plt.close()
        
        # Plot 3: Inference Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_pos, inference_times, alpha=0.7, color='orange')
        ax.set_xlabel("Variant")
        ax.set_ylabel("Inference Time (ms)")
        ax.set_title("Inference Speed Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variant_names, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/inference_time_comparison.png", dpi=150)
        plt.close()
        
        print(f"Plots saved to {output_dir}")


def get_model_size(model: torch.nn.Module) -> Tuple[float, int]:
    """
    Calculate model size and parameter count.
    
    Args:
        model: PyTorch model
    
    Returns:
        (size_mb, num_params)
    """
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / (1024 ** 2)
    
    return size_mb, num_params
