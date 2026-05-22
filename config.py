"""
Configuration file for LSTM + LLM Ablation Study Pipeline
"""

# ============================================
# Environment Config
# ============================================
ENV_NAME = "HalfCheetah-v4"
POMDP_OBS_DIM = 8  # qpos only (hide qvel)
FULL_STATE_DIM = 17  # qpos + qvel
ACTION_DIM = 6

# ============================================
# Training Config
# ============================================
NUM_EPISODES = 40
STEPS_PER_EPISODE = 1000

# Replay Buffer
REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 16
SEQ_LENGTH = 32  # Sequence length for LSTM

# Success Criteria
SUCCESS_RATE_THRESHOLD = 0.80
SUCCESS_EPISODE_REWARD = 1000  # Threshold for success in HalfCheetah

# ============================================
# Model Architecture Config
# ============================================
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1

# Student Network
STUDENT_FEAT_ENCODER_DIM = 64

# Teacher Network (LSTM only variant)
TEACHER_LSTM_FEAT_ENCODER_DIM = 128

# Teacher Network (with LLM variant)
LLM_LATENT_DIM = 256  # Dimension of LLM strategy vector
TEACHER_LLM_FEAT_ENCODER_DIM = 128

# ============================================
# PPO Hyperparameters (for Teacher Training)
# ============================================
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95  # GAE lambda
PPO_CLIP_RATIO = 0.2
PPO_NUM_EPOCHS = 3
PPO_NUM_MINIBATCHES = 2
PPO_ENTROPY_COEF = 0.01

# ============================================
# Distillation Training Config (Student)
# ============================================
DISTILLATION_LR = 1e-3
DISTILLATION_KL_WEIGHT = 1.0
DISTILLATION_MSE_WEIGHT = 0.5
DISTILLATION_NUM_EPOCHS = 5
DISTILLATION_LOSS_TYPE = "combined"  # "kl", "mse", "combined"

# ============================================
# LLM / Ollama Config
# ============================================
OLLAMA_MODEL = "qwen2:7b"  # qwen2:7b from ollama (smaller model)
OLLAMA_HOST = "localhost:11434"
OLLAMA_TIMEOUT = 10  # seconds
LLM_PROMPT_TEMPLATE = """
Current HalfCheetah state (position, velocity):
{state_info}

Based on this state, predict future action guidance.
Provide a high-level strategy (e.g., "accelerate forward", "stabilize balance", "turn left").
"""

# ============================================
# Evaluation & Metrics Config
# ============================================
EVAL_EPISODES = 5  # Per variant, at the end
EVAL_RENDER = False

METRICS_TO_LOG = [
    "episode_reward",
    "episode_length",
    "success",  # Boolean
    "inference_time_ms",
    "memory_usage_mb",
]

# ============================================
# Output & Logging Config
# ============================================
RESULTS_DIR = "./results/ablation_study"
CHECKPOINT_DIR = "./checkpoints"
LOG_INTERVAL = 5  # Log every N episodes

OUTPUT_FORMATS = ["csv", "markdown", "json"]
PLOT_TYPES = ["learning_curve", "reward_distribution", "success_rate_over_time"]

# ============================================
# Ablation Study Variants
# ============================================
VARIANTS = {
    "V1_LSTM_Only": {
        "name": "DL Baseline (LSTM only, no Teacher)",
        "use_teacher": False,
        "use_llm": False,
        "description": "Student LSTM learns standalone via PPO"
    },
    "V2_LSTM_Teacher": {
        "name": "With Teacher (LSTM only, no LLM)",
        "use_teacher": True,
        "use_llm": False,
        "description": "Student learns to mimic Teacher LSTM via distillation"
    },
    "V3_LSTM_LLM": {
        "name": "Full System (LSTM + LLM guidance)",
        "use_teacher": True,
        "use_llm": True,
        "description": "Student learns from Teacher (LSTM + LLM)"
    },
}

# ============================================
# Random Seed & Reproducibility
# ============================================
SEED = 42
DEVICE = "cuda"  # "cuda" or "cpu"
