# Usage Guide - LSTM + LLM Ablation Study Pipeline

## Quick Start (3 options)

### Option 1: Automatic Setup (Windows)

```bash
./quickstart.bat
```

This will guide you through setup and running options.

### Option 2: Manual Setup (All Platforms)

1. **Verify Setup**

```bash
python setup_check.py
```

2. **Run Quick Demo** (2-3 minutes)

```bash
python demo.py
```

3. **Run Full Ablation Study** (30-60 minutes)

```bash
python ablation_pipeline.py --variants all
```

### Option 3: Docker Setup

```bash
# Build image
docker build -t lstm-llm:latest .

# Run container
docker run -it --gpus all lstm-llm:latest python ablation_pipeline.py
```

---

## Step-by-Step Usage

### Step 1: Environment Setup

**Linux/Mac:**

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Windows:**

```bash
# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python setup_check.py
```

Expected output should show:

```
✓ Python 3.10.x
✓ torch (2.1.x)
✓ gymnasium
✓ All dependencies installed!
✓ All required files present!
✓ HalfCheetah environment working
```

### Step 3: Setup Ollama (for V3 variant)

**If you want to run all variants including V3 with LLM:**

**Using Docker (Recommended):**

```bash
# Terminal 1: Run Ollama
docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull gemma3

# Terminal 2: Run pipeline
python ablation_pipeline.py --variants all
```

**Local Installation:**

1. Download Ollama from https://ollama.ai
2. Install and run:

```bash
ollama serve
# In another terminal:
ollama pull gemma3
```

**Skip LLM (V1 + V2 only):**

```bash
python ablation_pipeline.py --variants V1 V2
```

### Step 4: Run the Ablation Study

**Run All Variants:**

```bash
python ablation_pipeline.py --variants all
```

**Run Specific Variants:**

```bash
# Only V1 (baseline)
python ablation_pipeline.py --variants V1

# Only V2 (with teacher)
python ablation_pipeline.py --variants V2

# Only V3 (with LLM)
python ablation_pipeline.py --variants V3

# V1 and V2 (no LLM needed)
python ablation_pipeline.py --variants V1 V2
```

**With Additional Options:**

```bash
# Render environment
python ablation_pipeline.py --variants all --render

# Custom results directory
python ablation_pipeline.py --results-dir ./my_results

# Combine options
python ablation_pipeline.py --variants V1 V2 --results-dir ./results2024
```

---

## Understanding the Results

### Output Files Structure

After running the pipeline, you'll find:

```
results/ablation_study/
│
├── V1_LSTM_Only/
│   ├── model.pt                    # Trained student model weights
│   ├── metrics.csv                 # Episode-by-episode metrics
│   └── metrics.json                # Full metrics in JSON format
│
├── V2_LSTM_Teacher/
│   ├── teacher.pt                  # Teacher LSTM model
│   ├── student.pt                  # Student LSTM model
│   ├── metrics.csv
│   └── metrics.json
│
├── V3_LSTM_LLM/
│   ├── teacher_llm.pt              # Teacher with LLM module
│   ├── student.pt                  # Student model
│   ├── metrics.csv
│   └── metrics.json
│
├── plots/
│   ├── reward_comparison.png       # Bar chart of mean rewards
│   ├── success_rate_comparison.png # Success rate comparison
│   └── inference_time_comparison.png # Speed comparison
│
└── ABLATION_COMPARISON.md          # Final analysis report (KEY FILE)
```

### Key Report: ABLATION_COMPARISON.md

This is the main output file. It contains:

**1. Summary Table**

```markdown
| Variant         | Mean Reward | Std Reward | Success Rate | Avg Inference |
| --------------- | ----------- | ---------- | ------------ | ------------- |
| V1_LSTM_Only    | 4523.45     | 523.23     | 45%          | 0.82 ms       |
| V2_LSTM_Teacher | 5234.12     | 312.11     | 72%          | 0.91 ms       |
| V3_LSTM_LLM     | 5891.34     | 234.56     | 88%          | 2.45 ms       |
```

**2. Detailed Analysis**

- Per-variant metrics breakdown
- Model size and parameter counts
- Performance summary

**3. Key Findings**

- Best performing variant
- Impact of Teacher (V2 vs V1)
- Impact of LLM (V3 vs V2)
- Computational overhead analysis

### Interpreting Metrics

| Metric        | Meaning                  | Good Value               |
| ------------- | ------------------------ | ------------------------ |
| Mean Reward   | Average episode reward   | Higher is better         |
| Std Reward    | Reward stability         | Lower is better (stable) |
| Success Rate  | % episodes > 1000 reward | 80%+ is good             |
| Avg Inference | Time per action decision | Lower is better          |
| Model Size    | Memory footprint         | Varies by variant        |

---

## Customization

### Modify Hyperparameters

Edit `config.py`:

```python
# Training
NUM_EPISODES = 40              # Increase for longer training
REPLAY_BUFFER_SIZE = 1000      # More trajectories = more stable
BATCH_SIZE = 16                # Smaller = more updates, slower

# Model Architecture
LSTM_HIDDEN_SIZE = 128         # Larger = more capacity
LLM_LATENT_DIM = 256          # LLM embedding dimension

# PPO Algorithm
PPO_LR = 3e-4                  # Learning rate
PPO_GAMMA = 0.99               # Discount factor (future importance)
PPO_CLIP_RATIO = 0.2           # Trust region width

# Distillation
DISTILLATION_LR = 1e-3         # Student learning rate
DISTILLATION_KL_WEIGHT = 1.0   # Action distribution matching
DISTILLATION_MSE_WEIGHT = 0.5  # Hidden state alignment
```

### Adjust Success Criteria

```python
# In config.py
SUCCESS_EPISODE_REWARD = 1000  # Threshold for "success"
SUCCESS_RATE_THRESHOLD = 0.80  # Target success rate
```

### Change Environment

The pipeline uses HalfCheetah-v4. To try other MuJoCo environments:

```python
# In config.py
ENV_NAME = "Walker2d-v4"  # or "Hopper-v4", "Ant-v4", etc.
```

Note: You may need to adjust observation/action dimensions.

---

## Common Use Cases

### Use Case 1: Validate Setup

```bash
python demo.py  # Takes 2-3 minutes
```

### Use Case 2: Quick Comparison (V1 vs V2)

```bash
python ablation_pipeline.py --variants V1 V2
```

Takes ~10-15 minutes.

### Use Case 3: Full Research Study

```bash
# Increase episodes in config.py
# NUM_EPISODES = 200

python ablation_pipeline.py --variants all
```

Takes 1-2 hours.

### Use Case 4: Reproduce Published Results

```bash
# Ensure seed is set in config.py
# SEED = 42

python ablation_pipeline.py --variants all --results-dir ./published_results
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'gymnasium'"

**Solution:**

```bash
pip install gymnasium[mujoco]
pip install mujoco
```

### Problem: Ollama Connection Error

```
Error: Could not connect to Ollama at http://localhost:11434
```

**Solution 1: Check Ollama is Running**

```bash
# Terminal 1
ollama serve

# Terminal 2
curl http://localhost:11434/api/tags
```

**Solution 2: Use Docker**

```bash
docker run -d -p 11434:11434 ollama/ollama
docker exec ollama ollama pull gemma3
```

**Solution 3: Skip LLM (Use V1 + V2 only)**

```bash
python ablation_pipeline.py --variants V1 V2
```

### Problem: Out of Memory Error

**Reduce batch size:**

```python
# In config.py
BATCH_SIZE = 8  # Instead of 16
LSTM_HIDDEN_SIZE = 64  # Instead of 128
```

### Problem: Training Too Slow

**Reduce training time:**

```python
# In config.py
NUM_EPISODES = 20  # Instead of 40
PPO_NUM_EPOCHS = 1  # Instead of 3
```

Or use CPU:

```bash
# In config.py
DEVICE = "cpu"
```

### Problem: CUDA Out of Memory

```bash
# Set device to CPU
# In config.py
DEVICE = "cpu"

# Or reduce model size
LSTM_HIDDEN_SIZE = 64
```

---

## Advanced: Custom Modifications

### Modify Model Architecture

Edit `models.py` to customize:

- Feature encoder dimensions
- LSTM num_layers
- Additional network modules

Example:

```python
# In models.py - StudentInferenceActor
self.feat_encoder = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),  # Add another layer
)
```

### Custom Loss Function

Edit `models.py` - `compute_distillation_loss()`:

```python
def compute_distillation_loss(...):
    # Customize loss components
    # Add weight decay, L1 regularization, etc.
    pass
```

### Track Additional Metrics

Edit `evaluation.py` - `PolicyEvaluator.evaluate()`:

```python
# Add custom metrics collection
# Save additional statistics
```

---

## Command Cheat Sheet

```bash
# Setup
conda activate lstm_llm
python setup_check.py

# Testing
python demo.py

# Run pipeline
python ablation_pipeline.py --variants all
python ablation_pipeline.py --variants V1 V2
python ablation_pipeline.py --results-dir ./custom_results

# View results
cat results/ablation_study/ABLATION_COMPARISON.md

# Cleanup
rm -rf results/ablation_study
```

---

## FAQ

**Q: How long does the full pipeline take?**
A: ~30-60 minutes depending on hardware (GPU vs CPU).

**Q: Can I run on CPU?**
A: Yes, set `DEVICE = "cpu"` in config.py, but it will be slow (~3-4x slower).

**Q: Do I need Ollama for V1 and V2?**
A: No, V3 uses LLM. V1 and V2 work without it.

**Q: Can I resume a crashed run?**
A: Not directly. Models are saved per variant. Re-run the pipeline.

**Q: How do I modify the RL algorithm?**
A: Edit `training.py` - `PPOTrainer` class methods.

**Q: Can I use different LLM models?**
A: Yes, change `OLLAMA_MODEL` in config.py (requires that model in Ollama).

---

## Next Steps

1. **Run the pipeline** and analyze results
2. **Modify hyperparameters** based on findings
3. **Implement improvements** to the architecture
4. **Experiment with different environments**
5. **Publish your findings!**

For questions or issues, check README.md or examine the code comments.
