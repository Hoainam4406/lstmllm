"""
Utility functions for LSTM + LLM Ablation Study Pipeline
Includes: Replay Buffer, logging, metrics collection
"""

import numpy as np
import torch
import csv
import json
from collections import deque
from pathlib import Path
from datetime import datetime
import psutil
import os


class ReplayBuffer:
    """Experience replay buffer for trajectory storage"""
    
    def __init__(self, max_size=1000, obs_dim=8, act_dim=6, seq_length=32, llm_dim=256):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seq_length = seq_length
        self.llm_dim = llm_dim
        
        # Buffers
        self.observations = deque(maxlen=max_size)
        self.full_states = deque(maxlen=max_size)  # Full state (17-dim) for Teacher
        self.llm_embeddings = deque(maxlen=max_size)  # LLM embeddings (256-dim) for V3
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.next_observations = deque(maxlen=max_size)
        
        self.current_trajectory_obs = []
        self.current_trajectory_full_state = []
        self.current_trajectory_llm_emb = []
        self.current_trajectory_actions = []
        self.current_trajectory_rewards = []
        self.current_trajectory_dones = []
        
    def add_step(self, obs, full_state, action, reward, done, next_obs, llm_z=None):
        """Add a single step to current trajectory"""
        self.current_trajectory_obs.append(obs)
        self.current_trajectory_full_state.append(full_state)
        self.current_trajectory_actions.append(action)
        self.current_trajectory_rewards.append(reward)
        self.current_trajectory_dones.append(done)
        # Store LLM embedding (zeros if not provided)
        if llm_z is not None:
            self.current_trajectory_llm_emb.append(llm_z)
        else:
            self.current_trajectory_llm_emb.append(np.zeros(self.llm_dim))
        
    def finalize_trajectory(self):
        """Finalize current trajectory and add to buffer"""
        if len(self.current_trajectory_obs) > 0:
            self.observations.append(np.array(self.current_trajectory_obs))
            self.full_states.append(np.array(self.current_trajectory_full_state))  # Store full state
            self.llm_embeddings.append(np.array(self.current_trajectory_llm_emb))  # Store LLM embeddings
            self.actions.append(np.array(self.current_trajectory_actions))
            self.rewards.append(np.array(self.current_trajectory_rewards))
            self.dones.append(np.array(self.current_trajectory_dones))
            
            # Reset trajectory buffers
            self.current_trajectory_obs = []
            self.current_trajectory_full_state = []
            self.current_trajectory_llm_emb = []
            self.current_trajectory_actions = []
            self.current_trajectory_rewards = []
            self.current_trajectory_dones = []
    
    def sample_batch(self, batch_size=16):
        """Sample a batch of trajectories"""
        if len(self.observations) < batch_size:
            batch_size = len(self.observations)
        
        indices = np.random.choice(len(self.observations), size=batch_size, replace=False)
        
        obs_batch = [self.observations[i] for i in indices]
        full_state_batch = [self.full_states[i] for i in indices]  # Get full states
        llm_emb_batch = [self.llm_embeddings[i] for i in indices]  # Get LLM embeddings
        act_batch = [self.actions[i] for i in indices]
        reward_batch = [self.rewards[i] for i in indices]
        
        # Pad/truncate to seq_length
        obs_seq = self._pad_sequences([torch.from_numpy(o).float() for o in obs_batch])
        full_state_seq = self._pad_sequences([torch.from_numpy(f).float() for f in full_state_batch])
        llm_emb_seq = self._pad_sequences([torch.from_numpy(e).float() for e in llm_emb_batch])
        act_seq = self._pad_sequences([torch.from_numpy(a).float() for a in act_batch])
        
        return obs_seq, full_state_seq, llm_emb_seq, act_seq, reward_batch
    
    def _pad_sequences(self, sequences):
        """Pad sequences to seq_length"""
        padded = []
        for seq in sequences:
            if len(seq) < self.seq_length:
                pad_len = self.seq_length - len(seq)
                seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[1])])
            else:
                seq = seq[:self.seq_length]
            padded.append(seq)
        return torch.stack(padded)
    
    def __len__(self):
        return len(self.observations)
    
    def clear(self):
        """Clear the buffer"""
        self.observations.clear()
        self.full_states.clear()
        self.llm_embeddings.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_observations.clear()


class MetricsLogger:
    """Log and track experiment metrics"""
    
    def __init__(self, variant_name, results_dir="./results"):
        self.variant_name = variant_name
        self.results_dir = Path(results_dir) / variant_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_metrics = []
        self.current_episode = {
            "episode": 0,
            "total_reward": 0.0,
            "episode_length": 0,
            "avg_action": None,
            "inference_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "success": False,
        }
        
    def log_episode_start(self, episode_num):
        self.current_episode = {
            "episode": episode_num,
            "total_reward": 0.0,
            "episode_length": 0,
            "inference_time_ms": [],
            "memory_usage_mb": 0.0,
            "steps": []
        }
    
    def log_step(self, reward, done, action=None, inference_time=None):
        """Log a single step within episode"""
        self.current_episode["total_reward"] += reward
        self.current_episode["episode_length"] += 1
        
        if inference_time is not None:
            self.current_episode["inference_time_ms"].append(inference_time)
        
        if action is not None:
            self.current_episode["steps"].append(float(np.linalg.norm(action)))
    
    def log_episode_end(self, success=False):
        """Finalize episode logging"""
        # Calculate averages
        if self.current_episode["inference_time_ms"]:
            self.current_episode["avg_inference_time"] = np.mean(
                self.current_episode["inference_time_ms"]
            )
        else:
            self.current_episode["avg_inference_time"] = 0.0
        
        self.current_episode["memory_usage_mb"] = self._get_memory_usage()
        self.current_episode["success"] = success
        
        self.episode_metrics.append(self.current_episode)
    
    def _get_memory_usage(self):
        """Get current process memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def save_metrics(self, formats=["csv", "json"]):
        """Save metrics to file"""
        for fmt in formats:
            if fmt == "csv":
                self._save_csv()
            elif fmt == "json":
                self._save_json()
    
    def _save_csv(self):
        """Save metrics as CSV"""
        csv_path = self.results_dir / "metrics.csv"
        if len(self.episode_metrics) > 0:
            keys = self.episode_metrics[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.episode_metrics)
    
    def _save_json(self):
        """Save metrics as JSON"""
        json_path = self.results_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.episode_metrics, f, indent=2, default=str)
    
    def get_summary(self):
        """Get summary statistics"""
        if len(self.episode_metrics) == 0:
            return {}
        
        rewards = [m["total_reward"] for m in self.episode_metrics]
        success_count = sum(1 for m in self.episode_metrics if m["success"])
        inference_times = [m["avg_inference_time"] for m in self.episode_metrics if "avg_inference_time" in m]
        
        return {
            "total_episodes": len(self.episode_metrics),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "success_rate": float(success_count / len(self.episode_metrics)),
            "mean_episode_length": float(np.mean([m["episode_length"] for m in self.episode_metrics])),
            "mean_inference_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
            "mean_memory_mb": float(np.mean([m["memory_usage_mb"] for m in self.episode_metrics])),
        }


def setup_experiment_dirs(results_base="./results"):
    """Create experiment directory structure"""
    results_path = Path(results_base)
    results_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = results_path / f"ablation_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    return exp_dir


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_str="cuda"):
    """Get torch device"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


def format_time(seconds):
    """Format seconds to readable time string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
