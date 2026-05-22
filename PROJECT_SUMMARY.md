# Project Summary - LSTM + LLM Guided RL Policy Ablation Study

## 📋 Overview

This project implements a complete **ablation study pipeline** comparing three variants of deep reinforcement learning policies:

1. **V1: LSTM Only (Baseline)** - Standard DL policy learning
2. **V2: LSTM + Teacher (No LLM)** - Knowledge distillation from Teacher
3. **V3: Full System (LSTM + LLM)** - LLM-guided strategy combined with LSTM

---

## 🗂️ Project Structure

### Core Files

| File                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| **ablation_pipeline.py** | Main entry point - runs all three experiment variants           |
| **config.py**            | Centralized configuration (hyperparameters, model sizes, paths) |
| **models.py**            | Neural network architectures (Student, Teacher, Teacher+LLM)    |
| **training.py**          | PPO trainer for Teacher networks                                |
| **distillation.py**      | Distillation trainer (Student learns from Teacher)              |
| **evaluation.py**        | Policy evaluation and comparison metrics                        |
| **llm_interface.py**     | Ollama integration (Gemma3 SLM)                                 |
| **utils.py**             | Utilities (replay buffer, logging, metrics)                     |
| **PO-MuJoCo.py**         | POMDP environment wrapper                                       |

### Documentation & Utilities

| File                 | Purpose                                   |
| -------------------- | ----------------------------------------- |
| **README.md**        | Project overview and architecture         |
| **USAGE_GUIDE.md**   | Detailed step-by-step usage instructions  |
| **setup_check.py**   | Verify dependencies and environment setup |
| **demo.py**          | Quick 2-minute demo to validate setup     |
| **quickstart.bat**   | Windows automated setup and launch        |
| **requirements.txt** | Python dependencies                       |

---

## 🎯 Key Components

### 1. Models (models.py)

**StudentInferenceActor**

- Observes partial state (POMDP: only positions, no velocities)
- Uses LSTM for sequential memory
- Outputs action distribution

**TeacherLSTMActor**

- Observes full state (positions + velocities)
- LSTM-based policy
- Used in V2 variant

**TeacherLLMActor**

- Observes full state + LLM guidance embedding
- Combines LSTM with LLM strategy vector
- Used in V3 variant

### 2. Training Algorithms

**PPO Trainer** (training.py)

- Proximal Policy Optimization
- Trains Teacher networks
- Collects trajectories, computes advantages, updates policy

**Distillation Trainer** (distillation.py)

- Knowledge distillation: Student learns from Teacher
- Loss = KL divergence (actions) + MSE (hidden states)
- Used in V2 and V3

**Independent Trainer** (distillation.py)

- Standalone policy learning (for V1)
- Direct policy gradient optimization

### 3. LLM Integration (llm_interface.py)

**OllamaInterface**

- Communicates with local Ollama server (Gemma3)
- Generates strategy embeddings from state
- Fallback to random embeddings if Ollama unavailable

**Features:**

- State parsing to natural language
- LLM-generated strategy guidance
- Deterministic text-to-embedding conversion

### 4. Evaluation & Comparison (evaluation.py)

**PolicyEvaluator**

- Runs policy on environment
- Collects metrics: reward, success rate, inference time, memory

**AblationStudyComparison**

- Compares all three variants
- Generates markdown report
- Creates comparison plots

---

## 📊 Metrics Collected

| Metric               | V1  | V2  | V3  |
| -------------------- | --- | --- | --- |
| Mean Episode Reward  | ✓   | ✓   | ✓   |
| Success Rate (%)     | ✓   | ✓   | ✓   |
| Episode Length       | ✓   | ✓   | ✓   |
| Inference Speed (ms) | ✓   | ✓   | ✓   |
| Model Size (MB)      | ✓   | ✓   | ✓   |
| Parameter Count      | ✓   | ✓   | ✓   |
| Learning Curve       | ✓   | ✓   | ✓   |

---

## 🚀 Quick Start

### 1. Verify Setup

```bash
python setup_check.py
```

### 2. Run Quick Demo (2-3 min)

```bash
python demo.py
```

### 3. Run Full Pipeline

```bash
python ablation_pipeline.py --variants all
```

### 4. View Results

Results saved to: `results/ablation_study/ABLATION_COMPARISON.md`

---

## 📈 Expected Results

### Typical Performance Pattern

| Variant | Reward             | Success Rate | Speed  | Why                                     |
| ------- | ------------------ | ------------ | ------ | --------------------------------------- |
| V1      | Low (3000-4000)    | ~30-50%      | Fast   | Learns from scratch, sample inefficient |
| V2      | Medium (4500-5500) | ~60-75%      | Medium | Teacher guidance improves learning      |
| V3      | High (5000-6500)   | ~75-90%      | Slower | LLM provides high-level strategy        |

### Key Insights

1. **Teacher Effect**: V2 > V1
   - Distillation provides inductive bias
   - Better sample efficiency
   - Faster convergence

2. **LLM Effect**: V3 > V2
   - LLM guidance improves final performance
   - Higher success rate
   - Trade-off: slower inference (LLM calls)

3. **Computational Cost**
   - V1: Baseline speed
   - V2: ~1.1x slower (teacher available during distillation)
   - V3: ~2-3x slower (LLM inference per action)

---

## 🔧 Configuration

Key hyperparameters in `config.py`:

```python
# Training
NUM_EPISODES = 40
REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 16

# Model
LSTM_HIDDEN_SIZE = 128
LLM_LATENT_DIM = 256

# Algorithm
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_CLIP_RATIO = 0.2

# LLM
OLLAMA_MODEL = "gemma3"
OLLAMA_HOST = "localhost:11434"

# Success
SUCCESS_EPISODE_REWARD = 1000
SUCCESS_RATE_THRESHOLD = 0.80
```

---

## 📦 Dependencies

### Core

- PyTorch >= 2.1.0 (with CUDA 12.1 support)
- Gymnasium[mujoco] >= 0.29.0
- NumPy >= 1.24.0

### Data & Visualization

- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

### Utilities

- Requests >= 2.31.0 (for Ollama API)
- psutil >= 5.9.0 (for memory monitoring)

### Optional

- Ollama (for V3 variant)
- Docker (for containerized Ollama)

---

## 🎓 How It Works

### Training Pipeline Flow

```
┌─────────────────┐
│  V1: LSTM Only  │
├─────────────────┤
│ 1. Init Student │
│ 2. Train (PPO)  │
│ 3. Evaluate     │
└─────────────────┘

┌─────────────────────────────────┐
│   V2: LSTM + Teacher            │
├─────────────────────────────────┤
│ 1. Init Student & Teacher       │
│ 2. Train Teacher (PPO)          │
│ 3. Collect Teacher Trajectories │
│ 4. Train Student (Distillation) │
│ 5. Evaluate                     │
└─────────────────────────────────┘

┌──────────────────────────────────────┐
│   V3: Full System (LSTM + LLM)      │
├──────────────────────────────────────┤
│ 1. Init Student, Teacher, LLM        │
│ 2. Train Teacher+LLM (PPO)          │
│ 3. Collect Teacher+LLM Trajectories │
│ 4. Train Student (Distillation)     │
│ 5. Evaluate (with LLM inference)    │
└──────────────────────────────────────┘
```

### Loss Functions

**V1: Policy Gradient**

```
Loss = -log_prob(action) + entropy_bonus
```

**V2 & V3: Distillation**

```
Loss = KL(Student_policy || Teacher_policy)
     + MSE(Student_hidden, Teacher_hidden)
```

---

## 📝 Output Files

### Per-Variant Output

```
results/ablation_study/V{1,2,3}_*/
├── model.pt or {teacher.pt, student.pt}
├── metrics.csv
└── metrics.json
```

### Comparison Output

```
results/ablation_study/
├── plots/
│   ├── reward_comparison.png
│   ├── success_rate_comparison.png
│   └── inference_time_comparison.png
└── ABLATION_COMPARISON.md  ← Main Report
```

---

## 🔍 Understanding the Report

The main output file `ABLATION_COMPARISON.md` contains:

1. **Summary Table** - Quick comparison of all metrics
2. **Detailed Results** - Per-variant breakdown
3. **Analysis** - Best performers, improvements, bottlenecks
4. **Conclusion** - Key findings and insights

Example findings:

- Which variant performs best
- Impact of Teacher distillation
- Impact of LLM guidance
- Computational overhead analysis

---

## 💡 How to Use This Project

### For Understanding RL + LLM

1. Read `models.py` to understand architecture
2. Read `training.py` and `distillation.py` for algorithms
3. Run `demo.py` to see training in action

### For Reproducing Results

1. Ensure `config.py` matches desired setup
2. Run `python ablation_pipeline.py --variants all`
3. Analyze `ABLATION_COMPARISON.md`

### For Research

1. Modify `config.py` for new hyperparameters
2. Modify `models.py` for new architectures
3. Run experiments and compare results

### For Deployment

1. Load saved models from checkpoint files
2. Use `PolicyEvaluator` for inference
3. Measure speed and memory on target hardware

---

## 🐛 Troubleshooting

| Problem                 | Solution                                  |
| ----------------------- | ----------------------------------------- |
| Missing dependencies    | `pip install -r requirements.txt`         |
| Ollama connection fails | Run `ollama serve` in another terminal    |
| Out of memory           | Reduce `BATCH_SIZE` or `LSTM_HIDDEN_SIZE` |
| Slow training           | Use GPU, reduce `NUM_EPISODES`            |
| MuJoCo not working      | `pip install gymnasium[mujoco] mujoco`    |

---

## 📚 References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Ollama](https://ollama.ai/)

---

## 📄 License

MIT License - See LICENSE file for details

---

## ✅ Checklist for First Run

- [ ] Install Python 3.10+
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python setup_check.py` (verify setup)
- [ ] Run `python demo.py` (quick test)
- [ ] Optionally setup Ollama for V3
- [ ] Run `python ablation_pipeline.py --variants all`
- [ ] Check `results/ablation_study/ABLATION_COMPARISON.md`
- [ ] View plots in `results/ablation_study/plots/`

---

Last Updated: 2024
Project Status: Ready for Production ✓
