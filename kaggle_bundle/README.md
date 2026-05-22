# LSTM + LLM Guided RL Policy - Ablation Study Pipeline

## Project Overview

This project implements an **ablation study** comparing three variants of deep reinforcement learning policies for the HalfCheetah locomotion task:

1. **V1: LSTM Only (Baseline)** - Student LSTM learns independently via policy gradient (PPO-like)
2. **V2: LSTM + Teacher (No LLM)** - Student LSTM learns to mimic a Teacher LSTM via knowledge distillation
3. **V3: Full System (LSTM + LLM)** - Student LSTM learns from a Teacher that combines LSTM with LLM-guided strategy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM + LLM Policy Learning                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  V1: LSTM Only               V2: LSTM + Teacher                │
│  ────────────────            ──────────────────                │
│      ┌─────────┐                  ┌──────────┐                 │
│      │ Student │                  │ Teacher  │                 │
│      │  LSTM   │◄─ PPO ────┐      │  LSTM    │                 │
│      └─────────┘            │      └──────────┘                 │
│           ▲                  │            ▲                      │
│           │                  │            │                      │
│      [Partial Obs]      [Distillation] [Full State]             │
│                                                                 │
│  V3: Full System (LSTM + LLM)                                  │
│  ───────────────────────────────                               │
│       ┌─────────┐         ┌──────────────┐                     │
│       │ Student │         │   Teacher    │                     │
│       │  LSTM   │◄─ KL+MSE│   LSTM       │                     │
│       └─────────┘         │   + LLM      │                     │
│           ▲                │ (Gemma3)     │                     │
│           │                └──────┬───────┘                     │
│      [Partial Obs]         [Full State + LLM Guidance]          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Models (`models.py`)

- **StudentInferenceActor**: Policy network that observes partial state (POMDP)
- **TeacherLSTMActor**: Teacher that observes full state
- **TeacherLLMActor**: Teacher with LLM-guided strategy embedding

### Training (`training.py`)

- **PPOTrainer**: Trains Teacher networks using Proximal Policy Optimization

### Distillation (`distillation.py`)

- **DistillationTrainer**: Trains Student to mimic Teacher via KL divergence + MSE loss
- **IndependentTrainer**: Trains Student independently (for V1)

### LLM Integration (`llm_interface.py`)

- **OllamaInterface**: Integrates with local Ollama (Gemma3)
- **MockOllamaInterface**: Fallback for when Ollama is not available

### Evaluation (`evaluation.py`)

- **PolicyEvaluator**: Evaluates policy performance (reward, success rate, latency, etc.)
- **AblationStudyComparison**: Compares variants and generates reports

## Setup & Installation

### 1. Create Virtual Environment

```bash
conda create -n lstm_llm python=3.10
conda activate lstm_llm
```

### 2. Install PyTorch with CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Ollama (for V3 variant)

**Option A: Using Docker** (Recommended)

```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull gemma3
```

**Option B: Local Installation**

```bash
# Download from https://ollama.ai
# Then run:
ollama serve &
ollama pull gemma3
```

**Option C: Command Line**

```bash
ollama run gemma3
```

## Configuration

Edit `config.py` to customize:

```python
# Training
NUM_EPISODES = 40              # Total episodes
REPLAY_BUFFER_SIZE = 1000      # Trajectory buffer size
BATCH_SIZE = 16                # Mini-batch size
SEQ_LENGTH = 32                # LSTM sequence length

# Architecture
LSTM_HIDDEN_SIZE = 128         # Hidden state dimension
LLM_LATENT_DIM = 256          # LLM embedding dimension

# PPO
PPO_LR = 3e-4                  # Learning rate
PPO_GAMMA = 0.99               # Discount factor
PPO_CLIP_RATIO = 0.2           # PPO clipping ratio

# LLM
OLLAMA_MODEL = "gemma3"
OLLAMA_HOST = "localhost:11434"

# Success criteria
SUCCESS_RATE_THRESHOLD = 0.80
SUCCESS_EPISODE_REWARD = 1000
```

## Running the Pipeline

### Run All Variants

```bash
python ablation_pipeline.py --variants all
```

### Run Specific Variants

```bash
# V1 only
python ablation_pipeline.py --variants V1

# V1 and V2
python ablation_pipeline.py --variants V1 V2

# V3 only (requires Ollama)
python ablation_pipeline.py --variants V3
```

### With Rendering

```bash
python ablation_pipeline.py --variants all --render
```

### Custom Results Directory

```bash
python ablation_pipeline.py --results-dir ./my_results
```

## Expected Output

### File Structure

```
results/ablation_study/
├── V1_LSTM_Only/
│   ├── model.pt              # Trained student model
│   └── metrics.csv           # Episode metrics
├── V2_LSTM_Teacher/
│   ├── teacher.pt            # Trained teacher
│   ├── student.pt            # Trained student
│   └── metrics.csv
├── V3_LSTM_LLM/
│   ├── teacher_llm.pt        # Teacher with LLM
│   ├── student.pt            # Student
│   └── metrics.csv
├── plots/
│   ├── reward_comparison.png
│   ├── success_rate_comparison.png
│   └── inference_time_comparison.png
└── ABLATION_COMPARISON.md    # Final report
```

### Report Example

The `ABLATION_COMPARISON.md` contains:

- Summary table comparing all variants
- Detailed metrics for each variant
- Best performing variant analysis
- Impact of Teacher and LLM

Example metrics:

```
| Variant              | Mean Reward | Success Rate | Inference (ms) |
|----------------------|-------------|--------------|----------------|
| V1_LSTM_Only         | 4523.45     | 45.0%        | 0.82 ms        |
| V2_LSTM_Teacher      | 5234.12     | 72.0%        | 0.91 ms        |
| V3_LSTM_LLM          | 5891.34     | 88.0%        | 2.45 ms        |
```

## Expected Results

Based on typical RL experiments:

1. **V1 (Baseline)**: Lower sample efficiency, learns from scratch
   - Mean Reward: ~4000-5000
   - Success Rate: 30-50%

2. **V2 (With Teacher)**: Better guidance through distillation
   - Mean Reward: ~5000-6000
   - Success Rate: 60-80%
   - Moderate speedup (~10-20%)

3. **V3 (Full System)**: Best performance with LLM guidance
   - Mean Reward: ~5500-7000
   - Success Rate: 75-95%
   - Small inference overhead (but better learning)

## Metrics Collected

For each variant:

- **Episode Reward**: Mean, std, min, max
- **Success Rate**: % episodes achieving reward > 1000
- **Episode Length**: Mean steps per episode
- **Inference Time**: Mean/std latency per action
- **Model Size**: Memory and parameter count
- **Learning Curve**: Reward over training iterations

## Troubleshooting

### Ollama Connection Fails

```
Error: Could not connect to Ollama at http://localhost:11434
```

**Solution:**

```bash
# Check if Ollama is running
ollama serve

# Or verify port
curl http://localhost:11434/api/tags
```

### Out of Memory

Reduce batch size in `config.py`:

```python
BATCH_SIZE = 8  # Instead of 16
```

### Slow Training

- Reduce `NUM_EPISODES`
- Reduce `REPLAY_BUFFER_SIZE`
- Use CPU: Set `DEVICE = "cpu"` (not recommended)

### LLM Timeouts

Increase timeout in `llm_interface.py`:

```python
self.timeout = 30  # Instead of 10
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lstm_llm_ablation,
  title={LSTM + LLM Guided Deep Reinforcement Learning for Policy Learning},
  year={2024}
}
```

## License

MIT License

## References

- [Gymnasium (OpenAI Gym)](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Ollama](https://ollama.ai/)
- [Gemma Model](https://deepmind.google/technologies/gemma/)
