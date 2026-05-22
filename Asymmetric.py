import torch
import torch.nn as nn
from torch.distributions import Normal

class StudentInferenceActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        # Encoder nén quan sát cục bộ
        self.feat_encoder = nn.Linear(obs_dim, 64)
        # Bộ nhớ tuần tự nội tại của Robot
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        # Output Action Distribution (Liên tục)
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs_seq, hidden_state=None):
        # obs_seq shape: (batch_size, seq_len, obs_dim)
        x = torch.relu(self.feat_encoder(obs_seq))
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        mu = self.mu_head(lstm_out)
        std = torch.exp(self.log_std_head).expand_as(mu)
        dist = Normal(mu, std)
        
        return dist, lstm_out, new_hidden

class TeacherLLMActor(nn.Module):
    def __init__(self, full_state_dim, llm_latent_dim, act_dim, hidden_size):
        super().__init__()
        # Giáo viên nhận cả State đầy đủ và vector ngữ nghĩa z từ LLM
        self.feat_encoder = nn.Linear(full_state_dim + llm_latent_dim, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, full_state_seq, llm_z_seq, hidden_state=None):
        # Nối (Concatenate) state và LLM latent
        x = torch.cat([full_state_seq, llm_z_seq], dim=-1)
        x = torch.relu(self.feat_encoder(x))
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        mu = self.mu_head(lstm_out)
        std = torch.exp(self.log_std_head).expand_as(mu)
        dist = Normal(mu, std)
        
        return dist, lstm_out, new_hidden