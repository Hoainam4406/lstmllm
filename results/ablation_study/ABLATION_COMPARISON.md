# LSTM + LLM Ablation Study Comparison

## Evaluation Results

| Variant | Mean Reward | Std Reward | Success Rate | Mean Length | Avg Inference (ms) | Model Size (MB) | Num Params |
|---|---|---|---|---|---|---|---|
| V1_LSTM_Only | -671.36 | 128.36 | 0.0% | 1000 | 1.53 | 0.38 | 100,684 |
| V2_LSTM_Teacher | -593.25 | 46.60 | 0.0% | 1000 | 1.83 | 0.38 | 100,684 |


## Detailed Results

### V1_LSTM_Only

**Evaluation Metrics:**
- Mean Episode Reward: -671.36 ± 128.36
- Max Reward: -548.45
- Min Reward: -866.79
- Success Rate (>1000): 0.0%
- Mean Episode Length: 1000
- Mean Inference Time: 1.53 ms
- Std Inference Time: 1.07 ms

**Model Info:**
- Model Size: 0.38 MB
- Number of Parameters: 100,684

### V2_LSTM_Teacher

**Evaluation Metrics:**
- Mean Episode Reward: -593.25 ± 46.60
- Max Reward: -528.88
- Min Reward: -654.85
- Success Rate (>1000): 0.0%
- Mean Episode Length: 1000
- Mean Inference Time: 1.83 ms
- Std Inference Time: 1.00 ms

**Model Info:**
- Model Size: 0.38 MB
- Number of Parameters: 100,684

## Analysis

**Best Mean Reward:** V2_LSTM_Teacher (-593.25)

**Best Success Rate:** V1_LSTM_Only (0.0%)

**Fastest Inference:** V1_LSTM_Only (1.53 ms)

## Conclusion

### Key Findings:
1. **Impact of Teacher (V2 vs V1):** 
2. **Impact of LLM (V3 vs V2):** 
3. **Computational Cost:** V2_LSTM_Teacher is 1.2x slower than V1_LSTM_Only.
