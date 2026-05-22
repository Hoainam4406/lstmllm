"""
PPO Trainer for Teacher Networks
Trains Teacher LSTM or Teacher LSTM+LLM using Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple, Optional, Dict, Any
import time

from models import TeacherLSTMActor, TeacherLLMActor
from llm_interface import OllamaInterface
from config import *


class PPOTrainer:
    """Trainer for Teacher networks using PPO algorithm"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        learning_rate: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        lambda_gae: float = PPO_LAMBDA,
        clip_ratio: float = PPO_CLIP_RATIO,
        num_epochs: int = PPO_NUM_EPOCHS,
        entropy_coef: float = PPO_ENTROPY_COEF,
        device: str = "cuda",
        use_llm: bool = False,
        llm_interface: Optional[OllamaInterface] = None,
    ):
        self.model = teacher_model.to(device)
        self.device = device
        self.use_llm = use_llm
        self.llm_interface = llm_interface
        
        # PPO hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Buffers for trajectory data
        self.reset_buffers()
    
    def reset_buffers(self):
        """Reset trajectory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def collect_trajectories(
        self,
        env: gym.Env,
        num_steps: int = 1000,
    ) -> Dict[str, float]:
        """
        Collect trajectories from environment using current policy.
        
        Args:
            env: Gymnasium environment
            num_steps: Number of environment steps to collect
        
        Returns:
            Episode statistics
        """
        self.reset_buffers()
        
        obs, _ = env.reset()
        hidden_state = None
        
        episode_rewards = []
        current_episode_reward = 0.0
        step_count = 0
        
        while step_count < num_steps:
            # Convert observation to tensor
            if isinstance(obs, np.ndarray):
                state_tensor = torch.from_numpy(obs).float().to(self.device)
            else:
                state_tensor = obs.to(self.device)
            
            # Get LLM embedding if needed
            llm_z = None
            if self.use_llm and self.llm_interface:
                llm_z = self.llm_interface.generate_strategy_embedding(
                    obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
                ).to(self.device)
            
            # Policy forward pass
            with torch.no_grad():
                if self.use_llm and llm_z is not None:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
                    llm_z = llm_z.unsqueeze(0).unsqueeze(0)  # (1, 1, llm_dim)
                    dist, _, hidden_state = self.model(state_tensor, llm_z, hidden_state)
                else:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
                    dist, _, hidden_state = self.model(state_tensor, hidden_state)
                
                # Sample action
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Environment step
            action_np = action.squeeze().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.states.append(obs)
            self.actions.append(action_np)
            self.rewards.append(reward)
            self.dones.append(done)
            self.log_probs.append(log_prob.cpu().item())
            
            current_episode_reward += reward
            step_count += 1
            
            obs = next_obs
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()
                hidden_state = None
        
        return {
            "num_episodes": len(episode_rewards),
            "total_steps": step_count,
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "std_episode_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        }
    
    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: Computed advantages
            returns: Target returns
        """
        num_steps = len(self.rewards)
        advantages = np.zeros(num_steps)
        returns = np.zeros(num_steps)
        
        next_value = 0
        gae = 0
        
        # Backward pass through trajectory
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = 0  # Bootstrap value
            else:
                next_nonterminal = 1.0 - self.dones[t]
                # In practice, compute value from model
                next_value = 0  # Simplified
            
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - 0
            gae = delta + self.gamma * self.lambda_gae * next_nonterminal * gae
            
            returns[t] = gae
            advantages[t] = gae
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def train_epoch(self) -> float:
        """
        Train one PPO epoch on collected trajectories.
        
        Returns:
            Average loss over the epoch
        """
        advantages, returns = self.compute_advantages()
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        
        # Convert buffers to tensors
        states_tensor = self._build_state_tensor()
        actions_tensor = torch.from_numpy(np.array(self.actions)).float().to(self.device)
        old_log_probs = torch.from_numpy(np.array(self.log_probs)).float().to(self.device)
        
        total_loss = 0.0
        num_updates = 0
        
        # Mini-batch updates
        for epoch in range(self.num_epochs):
            batch_size = len(self.states) // PPO_NUM_MINIBATCHES
            indices = np.random.permutation(len(self.states))
            
            for i in range(0, len(self.states), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Policy forward pass
                if self.use_llm and self.llm_interface:
                    # Generate LLM embeddings for batch
                    batch_size = batch_states.shape[0]
                    seq_len = batch_states.shape[1]
                    
                    llm_embeddings = []
                    for i in range(batch_size):
                        # Get first state in sequence for this batch item
                        state_np = batch_states[i, 0].cpu().numpy()
                        llm_z = self.llm_interface.generate_strategy_embedding(state_np)
                        # Repeat embedding across sequence length
                        llm_z_seq = llm_z.unsqueeze(0).repeat(seq_len, 1)  # (seq_len, 256)
                        llm_embeddings.append(llm_z_seq)
                    
                    llm_batch = torch.stack(llm_embeddings).to(self.device)  # (batch, seq_len, 256)
                    dist, _, _ = self.model(batch_states, llm_batch)
                else:
                    dist, _, _ = self.model(batch_states)
                
                # PPO loss
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clip loss
                clip_loss = -torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                ).mean()
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                loss = clip_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / (num_updates + 1e-8)
    
    def _build_state_tensor(self) -> torch.Tensor:
        """Convert state buffer to tensor"""
        states = []
        for state in self.states:
            if isinstance(state, np.ndarray):
                states.append(torch.from_numpy(state).float())
            else:
                states.append(state.float())
        
        return torch.stack(states).to(self.device)
    
    def train(
        self,
        env: gym.Env,
        num_updates: int = 10,
        steps_per_update: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train teacher for multiple PPO updates.
        
        Args:
            env: Environment
            num_updates: Number of PPO update rounds
            steps_per_update: Steps to collect per update
        
        Returns:
            Training statistics
        """
        train_stats = {
            "update": [],
            "mean_reward": [],
            "mean_loss": [],
        }
        
        for update in range(num_updates):
            # Collect trajectories
            traj_stats = self.collect_trajectories(env, steps_per_update)
            
            # Train on collected data
            loss = self.train_epoch()
            
            train_stats["update"].append(update)
            train_stats["mean_reward"].append(traj_stats["mean_episode_reward"])
            train_stats["mean_loss"].append(loss)
            
            if (update + 1) % 1 == 0:
                print(f"Update {update+1}/{num_updates} | "
                      f"Reward: {traj_stats['mean_episode_reward']:.2f} | "
                      f"Loss: {loss:.4f}")
        
        return train_stats
    
    def save(self, filepath: str):
        """Save trained model"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
