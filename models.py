"""
LSTM + LLM Policy Network Models
Includes: StudentInferenceActor, TeacherLSTMActor, TeacherLLMActor
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


class StudentInferenceActor(nn.Module):
    """
    Student policy network.
    Observes only partial state (qpos, no qvel) and learns to mimic Teacher.
    Uses LSTM for memory of past observations.
    """
    
    def __init__(
        self,
        obs_dim: int = 8,
        act_dim: int = 6,
        hidden_size: int = 128,
        feat_encoder_dim: int = 64,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        
        # Feature encoder: compress observation
        self.feat_encoder = nn.Linear(obs_dim, feat_encoder_dim)
        
        # LSTM: sequential memory
        self.lstm = nn.LSTM(
            input_size=feat_encoder_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Action distribution heads
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, act_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Normal, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs_seq: (batch_size, seq_len, obs_dim)
            hidden_state: Optional (h, c) from previous step
        
        Returns:
            dist: Normal distribution over actions
            lstm_out: LSTM output (batch_size, seq_len, hidden_size)
            new_hidden: Updated (h, c) state
        """
        # Encode observations
        x = torch.relu(self.feat_encoder(obs_seq))  # (batch, seq_len, feat_dim)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)  # (batch, seq_len, hidden_size)
        
        # Action distribution
        mu = self.mu_head(lstm_out)  # (batch, seq_len, act_dim)
        std = torch.exp(self.log_std_head).expand_as(mu)
        
        dist = Normal(mu, std)
        
        return dist, lstm_out, new_hidden
    
    def get_action(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample action from policy.
        
        Args:
            obs: Single observation (obs_dim,) or sequence
            hidden_state: Optional previous hidden state
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            new_hidden: Updated hidden state
        """
        # Handle both single and batched input
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0)  # (1, seq_len, obs_dim)
        
        dist, _, new_hidden = self.forward(obs, hidden_state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0), log_prob, new_hidden


class TeacherLSTMActor(nn.Module):
    """
    Teacher policy network (LSTM only variant).
    Observes full state (qpos + qvel) and generates actions.
    Used for behavioral cloning / distillation.
    """
    
    def __init__(
        self,
        full_state_dim: int = 17,
        act_dim: int = 6,
        hidden_size: int = 128,
        feat_encoder_dim: int = 128,
    ):
        super().__init__()
        
        self.full_state_dim = full_state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        
        # Feature encoder
        self.feat_encoder = nn.Linear(full_state_dim, feat_encoder_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=feat_encoder_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Action distribution heads
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, act_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        full_state_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Normal, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            full_state_seq: (batch_size, seq_len, full_state_dim)
            hidden_state: Optional (h, c) from previous step
        
        Returns:
            dist: Normal distribution over actions
            lstm_out: LSTM output
            new_hidden: Updated hidden state
        """
        x = torch.relu(self.feat_encoder(full_state_seq))
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        mu = self.mu_head(lstm_out)
        std = torch.exp(self.log_std_head).expand_as(mu)
        
        dist = Normal(mu, std)
        
        return dist, lstm_out, new_hidden


class TeacherLLMActor(nn.Module):
    """
    Teacher policy network with LLM guidance.
    Observes full state + LLM strategy embedding and generates actions.
    The LLM embedding provides high-level task understanding.
    """
    
    def __init__(
        self,
        full_state_dim: int = 17,
        llm_latent_dim: int = 256,
        act_dim: int = 6,
        hidden_size: int = 128,
        feat_encoder_dim: int = 128,
    ):
        super().__init__()
        
        self.full_state_dim = full_state_dim
        self.llm_latent_dim = llm_latent_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        
        # Feature encoder: combines state and LLM embedding
        input_dim = full_state_dim + llm_latent_dim
        self.feat_encoder = nn.Sequential(
            nn.Linear(input_dim, feat_encoder_dim),
            nn.ReLU(),
            nn.Linear(feat_encoder_dim, feat_encoder_dim),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=feat_encoder_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Action distribution heads
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, act_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        full_state_seq: torch.Tensor,
        llm_z_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Normal, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            full_state_seq: (batch_size, seq_len, full_state_dim)
            llm_z_seq: (batch_size, seq_len, llm_latent_dim)
            hidden_state: Optional (h, c) from previous step
        
        Returns:
            dist: Normal distribution over actions
            lstm_out: LSTM output
            new_hidden: Updated hidden state
        """
        # Concatenate state and LLM embedding
        x = torch.cat([full_state_seq, llm_z_seq], dim=-1)  # (batch, seq_len, input_dim)
        
        # Encode combined input
        x = self.feat_encoder(x)  # (batch, seq_len, feat_encoder_dim)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)  # (batch, seq_len, hidden_size)
        
        # Action distribution
        mu = self.mu_head(lstm_out)  # (batch, seq_len, act_dim)
        std = torch.exp(self.log_std_head).expand_as(mu)
        
        dist = Normal(mu, std)
        
        return dist, lstm_out, new_hidden


# Loss functions for training

def compute_distillation_loss(
    student_dist: Normal,
    teacher_dist: Normal,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    kl_weight: float = 1.0,
    mse_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute distillation loss combining KL divergence and MSE on hidden states.
    
    Args:
        student_dist: Student action distribution
        teacher_dist: Teacher action distribution
        student_hidden: Student LSTM hidden output
        teacher_hidden: Teacher LSTM hidden output
        kl_weight: Weight for KL divergence loss
        mse_weight: Weight for MSE loss
    
    Returns:
        Combined loss tensor
    """
    # KL divergence between action distributions
    kl_loss = torch.distributions.kl.kl_divergence(student_dist, teacher_dist).mean()
    
    # MSE loss on hidden states (representation learning)
    mse_loss = nn.MSELoss()(student_hidden, teacher_hidden.detach())
    
    # Combined loss
    total_loss = kl_weight * kl_loss + mse_weight * mse_loss
    
    return total_loss


def compute_ppo_loss(
    policy_dist: Normal,
    old_dist: Normal,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    Compute PPO loss for policy gradient training.
    
    Args:
        policy_dist: Current policy distribution
        old_dist: Old policy distribution (before update)
        advantages: Computed advantages
        clip_ratio: PPO clipping ratio
        entropy_coef: Entropy bonus coefficient
    
    Returns:
        PPO loss tensor
    """
    # Log probability ratio
    log_ratio = policy_dist.log_prob(old_dist.sample()) - old_dist.log_prob(old_dist.sample())
    ratio = torch.exp(log_ratio)
    
    # PPO clip loss
    clip_loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    ).mean()
    
    # Entropy bonus
    entropy_loss = -policy_dist.entropy().mean()
    
    return clip_loss + entropy_coef * entropy_loss
