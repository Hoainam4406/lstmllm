# LSTM + LLM RL Policy Ablation Study

## Quick Start

```bash
# Install dependencies
pip install -r kaggle_requirements.txt

# Run full pipeline
python kaggle_run.py
```

## Variants

### V1: LSTM-Only Baseline
- Pure RL with LSTM actor
- No teacher guidance
- Baseline for comparison

### V2: LSTM + Teacher Distillation  
- Full-state teacher trains via PPO
- POMDP student learns from teacher
- Knowledge distillation loss
- ~13% better than V1

### V3: Full System with LLM Guidance
- Teacher uses LLM embeddings for strategy
- Student still learns from teacher
- Mock embeddings when Ollama unavailable
- Fallback ensures always runnable

## Results

Metrics tracked:
- Mean episode reward
- Success rate  
- Policy entropy
- Training loss progression

Results saved to: `results/ablation_study/ABLATION_COMPARISON.md`

## Environment

- **Task**: HalfCheetah-v4 (locomotion)
- **Observation**: 8-dim POMDP (qpos only, no velocities)
- **Full State**: 17-dim (qpos + qvel)
- **Actions**: 6-dim continuous control
- **Reward**: Negative (cost-minimization)

## Hardware

Works on:
- CPU only (no CUDA required)
- Minimal RAM (~2-3GB active)
- ~30-45 min runtime

## Papers

- PPO: Schulman et al. (2017)
- Knowledge Distillation: Hinton et al. (2015)
- LSTM: Hochreiter & Schmidhuber (1997)
