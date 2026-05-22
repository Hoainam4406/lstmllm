"""
Distillation Training for Student Networks
Student learns to mimic Teacher via knowledge distillation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any
import time

from models import StudentInferenceActor, TeacherLSTMActor, TeacherLLMActor
from models import compute_distillation_loss
from utils import ReplayBuffer, MetricsLogger
from llm_interface import OllamaInterface
from config import *


class DistillationTrainer:
    """Trainer for Student networks via knowledge distillation from Teacher"""
    
    def __init__(
        self,
        student_model: StudentInferenceActor,
        teacher_model: nn.Module,
        student_lr: float = DISTILLATION_LR,
        kl_weight: float = DISTILLATION_KL_WEIGHT,
        mse_weight: float = DISTILLATION_MSE_WEIGHT,
        device: str = "cuda",
        use_llm: bool = False,
        llm_interface: Optional[OllamaInterface] = None,
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.use_llm = use_llm
        self.llm_interface = llm_interface
        
        # Loss weights
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        
        # Optimizer
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=student_lr)
        
        # Replay buffer for trajectory collection
        self.buffer = ReplayBuffer(
            max_size=REPLAY_BUFFER_SIZE,
            obs_dim=POMDP_OBS_DIM,
            act_dim=ACTION_DIM,
            seq_length=SEQ_LENGTH,
            llm_dim=256,  # LLM embedding dimension
        )
    
    def collect_trajectories_from_teacher(
        self,
        env_teacher: gym.Env,
        env_student: gym.Env,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Collect trajectories from Teacher policy.
        These will be used to train Student.
        
        Args:
            env_teacher: Environment with full state observations (17-dim)
            env_student: Environment with POMDP observations (8-dim)  
            num_episodes: Number of episodes to collect
        
        Returns:
            Collection statistics
        """
        self.buffer.clear()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            full_state, _ = env_teacher.reset()  # Get full 17-dim state from teacher env
            obs, _ = env_student.reset()  # Get POMDP 8-dim from student env
            
            teacher_hidden = None
            episode_reward = 0.0
            episode_length = 0
            
            done = False
            while not done:
                # Get observation (partial, as Student sees)
                pomdp_obs = obs[:POMDP_OBS_DIM]  # Only qpos (8-dim)
                
                # Teacher policy forward pass
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(full_state).float().to(self.device)
                    obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, full_state_dim=17)
                    
                    if self.use_llm and self.llm_interface:
                        llm_z = self.llm_interface.generate_strategy_embedding(obs).to(self.device)
                        llm_z = llm_z.unsqueeze(0).unsqueeze(0)  # (1, 1, llm_dim)
                        dist, _, teacher_hidden = self.teacher(obs_tensor, llm_z, teacher_hidden)
                    else:
                        dist, _, teacher_hidden = self.teacher(obs_tensor, teacher_hidden)
                    
                    # Sample action from Teacher
                    action = dist.sample()
                
                # Environment step
                action_np = action.squeeze().cpu().numpy()
                next_obs_student, reward, terminated, truncated, info = env_student.step(action_np)
                next_full_state, _, _, _, _ = env_teacher.step(action_np)
                done = terminated or truncated
                
                # Get LLM embedding if using V3
                llm_z_np = None
                if self.use_llm and self.llm_interface:
                    llm_z_np = llm_z.squeeze().cpu().numpy()
                
                # Store in buffer
                self.buffer.add_step(
                    obs=pomdp_obs,
                    full_state=full_state,
                    action=action_np,
                    reward=reward,
                    done=done,
                    next_obs=next_obs_student[:POMDP_OBS_DIM],
                    llm_z=llm_z_np
                )
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs_student
                full_state = next_full_state
            
            self.buffer.finalize_trajectory()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
        }
    
    def train_on_batch(self) -> float:
        """
        Train student on a batch of trajectories from replay buffer.
        
        Returns:
            Loss value
        """
        if len(self.buffer) == 0:
            return 0.0
        
        # Sample batch (now includes llm_emb_batch)
        obs_batch, full_state_batch, llm_emb_batch, act_batch, reward_batch = self.buffer.sample_batch(batch_size=BATCH_SIZE)
        
        obs_batch = obs_batch.to(self.device)  # (batch, seq_len, 8)
        full_state_batch = full_state_batch.to(self.device)  # (batch, seq_len, 17)
        act_batch = act_batch.to(self.device)  # (batch, seq_len, act_dim)
        
        # Student forward pass
        student_dist, student_hidden, _ = self.student(obs_batch)
        
        # Teacher forward pass (no gradient)
        with torch.no_grad():
            if self.use_llm and self.llm_interface:
                # Use pre-computed LLM embeddings from buffer
                llm_batch = llm_emb_batch.to(self.device)
                teacher_dist, teacher_hidden, _ = self.teacher(full_state_batch, llm_batch, None)
            else:
                teacher_dist, teacher_hidden, _ = self.teacher(full_state_batch, None)
        
        # Distillation loss
        loss = compute_distillation_loss(
            student_dist=student_dist,
            teacher_dist=teacher_dist,
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            kl_weight=self.kl_weight,
            mse_weight=self.mse_weight,
        )
        
        # Optimize student
        self.student_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.student_optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, num_batches: int = 50) -> float:
        """
        Train for one epoch (multiple batch updates).
        
        Args:
            num_batches: Number of batches to train on
        
        Returns:
            Average loss
        """
        total_loss = 0.0
        
        for _ in range(num_batches):
            loss = self.train_on_batch()
            total_loss += loss
        
        return total_loss / (num_batches + 1e-8)
    
    def train(
        self,
        env_student: gym.Env,
        env_teacher: gym.Env,
        num_iterations: int = 5,
        collection_episodes: int = 5,
        batches_per_iteration: int = 50,
    ) -> Dict[str, Any]:
        """
        Train student via distillation from teacher.
        
        Args:
            env_student: Environment with POMDP observations (8-dim)
            env_teacher: Environment with full state (17-dim)
            num_iterations: Number of iterations (collect + train)
            collection_episodes: Episodes to collect per iteration
            batches_per_iteration: Batches to train on per iteration
        
        Returns:
            Training statistics
        """
        train_stats = {
            "iteration": [],
            "teacher_reward": [],
            "train_loss": [],
        }
        
        for iteration in range(num_iterations):
            # Collect trajectories from Teacher
            print(f"\n[Iteration {iteration+1}/{num_iterations}] Collecting trajectories from Teacher...")
            coll_stats = self.collect_trajectories_from_teacher(env_teacher, env_student, collection_episodes)
            
            # Train Student
            print(f"Training Student on {BATCH_SIZE} batches...")
            loss = self.train_epoch(batches_per_iteration)
            
            train_stats["iteration"].append(iteration)
            train_stats["teacher_reward"].append(coll_stats["mean_reward"])
            train_stats["train_loss"].append(loss)
            
            print(f"  Teacher Reward: {coll_stats['mean_reward']:.2f} | Loss: {loss:.4f}")
        
        return train_stats
    
    def save(self, filepath: str):
        """Save trained student model"""
        torch.save(self.student.state_dict(), filepath)
        print(f"Student model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained student model"""
        self.student.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Student model loaded from {filepath}")


class IndependentTrainer:
    """
    Train Student independently without a Teacher.
    Uses PPO-like algorithm directly on Student.
    """
    
    def __init__(
        self,
        student_model: StudentInferenceActor,
        lr: float = PPO_LR,
        device: str = "cuda",
    ):
        self.student = student_model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)
        self.buffer = ReplayBuffer(
            max_size=REPLAY_BUFFER_SIZE,
            obs_dim=POMDP_OBS_DIM,
            act_dim=ACTION_DIM,
            seq_length=SEQ_LENGTH,
        )
    
    def collect_trajectories(
        self,
        env: gym.Env,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Collect trajectories using current Student policy.
        
        Args:
            env: Environment
            num_episodes: Number of episodes
        
        Returns:
            Collection statistics
        """
        self.buffer.clear()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            
            student_hidden = None
            episode_reward = 0.0
            done = False
            
            while not done:
                # Get partial observation (POMDP)
                pomdp_obs = obs[:POMDP_OBS_DIM]
                
                # Student policy
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(pomdp_obs).float().to(self.device)
                    obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                    
                    dist, _, student_hidden = self.student(obs_tensor, student_hidden)
                    action = dist.sample()
                
                # Environment step
                action_np = action.squeeze().cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                
                # Store
                self.buffer.add_step(
                    obs=pomdp_obs,
                    full_state=obs,
                    action=action_np,
                    reward=reward,
                    done=done,
                    next_obs=next_obs[:POMDP_OBS_DIM]
                )
                
                episode_reward += reward
                obs = next_obs
            
            self.buffer.finalize_trajectory()
            episode_rewards.append(episode_reward)
        
        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
        }
    
    def train_on_batch(self) -> float:
        """Train student on batch via policy gradient"""
        if len(self.buffer) == 0:
            return 0.0
        
        # sample_batch now returns 5 values (including llm_emb_batch)
        obs_batch, _, llm_emb_batch, act_batch, _ = self.buffer.sample_batch(batch_size=BATCH_SIZE)
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        
        # Student forward pass
        dist, _, _ = self.student(obs_batch)
        
        # Policy gradient loss (simple version: maximize log prob of taken actions)
        log_prob = dist.log_prob(act_batch).sum(dim=-1)
        loss = -log_prob.mean()  # Negative because we want to maximize
        
        # Add entropy bonus
        entropy = dist.entropy().mean()
        loss = loss - PPO_ENTROPY_COEF * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        env: gym.Env,
        num_iterations: int = 10,
        episodes_per_iteration: int = 5,
        batches_per_iteration: int = 50,
    ) -> Dict[str, Any]:
        """
        Train student independently.
        
        Args:
            env: Environment
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            batches_per_iteration: Batches to train on
        
        Returns:
            Training statistics
        """
        train_stats = {
            "iteration": [],
            "reward": [],
            "loss": [],
        }
        
        for iteration in range(num_iterations):
            print(f"\n[Iteration {iteration+1}/{num_iterations}] Collecting trajectories...")
            coll_stats = self.collect_trajectories(env, episodes_per_iteration)
            
            print(f"Training Student...")
            total_loss = 0.0
            for _ in range(batches_per_iteration):
                loss = self.train_on_batch()
                total_loss += loss
            
            avg_loss = total_loss / batches_per_iteration
            
            train_stats["iteration"].append(iteration)
            train_stats["reward"].append(coll_stats["mean_reward"])
            train_stats["loss"].append(avg_loss)
            
            print(f"  Reward: {coll_stats['mean_reward']:.2f} | Loss: {avg_loss:.4f}")
        
        return train_stats
    
    def save(self, filepath: str):
        """Save trained model"""
        torch.save(self.student.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        self.student.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
