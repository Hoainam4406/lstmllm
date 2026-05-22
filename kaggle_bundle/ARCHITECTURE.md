# Architecture & Data Flow Diagrams

## 1. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LSTM + LLM RL Policy Learning               │
│                      Ablation Study Pipeline                     │
└──────────────────────────────────────────────────────────────────┘

                              ┌─────────┐
                              │Environment
                              │(HalfCheetah)
                              └────┬────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
                  ┌──┐           ┌──┐           ┌──┐
                  │V1│           │V2│           │V3│
                  └──┘           └──┘           └──┘
                   │              │              │
        ┌──────────┘              │              └──────────┐
        │                         │                         │
        ▼                         ▼                         ▼
    ┌────────┐          ┌──────────────┐        ┌──────────────────┐
    │Student │          │ Teacher+     │        │  Teacher+        │
    │ LSTM   │          │ Student      │        │  LLM+            │
    │(only)  │          │ (LSTM)       │        │  Student (LSTM)  │
    └────────┘          └──────────────┘        └──────────────────┘
        │                     │                          │
        │                     ▼                          ▼
        │              ┌────────────────┐     ┌──────────────────┐
        └─────────────►│  Metrics &     │────►│  Report          │
                       │  Comparison    │     │  Generation      │
                       └────────────────┘     └──────────────────┘
                                                      │
                                                      ▼
                                         ┌──────────────────────┐
                                         │ABLATION_COMPARISON.md│
                                         │  (Main Result File)  │
                                         └──────────────────────┘
```

## 2. V1: LSTM Only (Baseline)

```
┌────────────────────────────────────────────────────────────────┐
│                        V1: LSTM Only (Baseline)                │
└────────────────────────────────────────────────────────────────┘

   Environment
       │
       │ Partial State (POMDP)
       │ [qpos only, 8-dim]
       ▼
   ┌─────────────────────────┐
   │ Student LSTM Actor      │
   │ ┌─────────────┐         │
   │ │ Feat        │         │
   │ │ Encoder     │         │
   │ │ (8→64)      │         │
   │ └──────┬──────┘         │
   │        │                │
   │ ┌──────▼──────┐         │
   │ │ LSTM        │         │
   │ │ hidden=128  │         │
   │ └──────┬──────┘         │
   │        │                │
   │ ┌──────▼──────┐         │
   │ │ Action Dist │         │
   │ │ Head        │         │
   │ └──────┬──────┘         │
   │        │                │
   └────────┼────────────────┘
            │
            ├─► Log Prob
            │
            ├─► Action Distribution
            │
            └─► Entropy
                     │
                     ▼
            ┌──────────────────┐
            │ Policy Gradient  │
            │ Loss             │
            │                  │
            │ L = -log_prob +  │
            │     entropy      │
            └──────────────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Backprop &       │
            │ Update           │
            └──────────────────┘
```

**Training Flow:**

1. Collect trajectories (40 episodes × 1000 steps)
2. Compute advantages (GAE)
3. Update policy with PPO loss
4. Repeat for multiple epochs

**Characteristics:**

- ✓ Simple, direct learning
- ✗ Lower sample efficiency
- ✓ Fast training
- ✗ Cannot leverage expert information

---

## 3. V2: LSTM + Teacher (No LLM)

```
┌────────────────────────────────────────────────────────────────┐
│           V2: LSTM + Teacher (Knowledge Distillation)          │
└────────────────────────────────────────────────────────────────┘

   Environment
       │
       ├─► Partial State (POMDP)      Full State (Complete)
       │   [qpos only, 8-dim]         [qpos + qvel, 17-dim]
       │                              │
       ▼                              ▼
   ┌─────────────────────┐      ┌──────────────────┐
   │ Student LSTM        │      │ Teacher LSTM     │
   │ (Learns)            │      │ (Expert Policy)  │
   │                     │      │                  │
   │ [Same arch as V1]   │      │ ┌──────────────┐ │
   │                     │      │ │ Feat Encoder │ │
   │                     │      │ │ (17→128)     │ │
   │                     │      │ └────┬─────────┘ │
   │                     │      │      │           │
   │                     │      │ ┌────▼────────┐  │
   │                     │      │ │ LSTM        │  │
   │                     │      │ │ hidden=128  │  │
   │                     │      │ └────┬────────┘  │
   │                     │      │      │           │
   │                     │      │ ┌────▼────────┐  │
   │                     │      │ │ Action Dist │  │
   │                     │      │ │ Head        │  │
   │                     │      │ └────┬────────┘  │
   │                     │      │      │           │
   │                     │      └──────┼───────────┘
   │                     │             │
   │                     │      [No Gradient]
   │                     │             │
   │                     ├─────────────┤
   │                     │             │
   │  dist              │  teacher_dist
   │  hidden            │  teacher_hidden
   │  ▼                 │  ◄─────────────┘
   │┌──────────────────────────────────┐
   ││  DISTILLATION LOSS               │
   ││                                  │
   ││  L = KL(S_dist || T_dist) +      │
   ││      MSE(S_hidden, T_hidden)     │
   │└──────────────────────────────────┘
   │  │
   │  ▼
   │┌──────────────────────────────────┐
   ││ Backprop to Student Only         │
   │└──────────────────────────────────┘
```

**Two-Phase Training:**

Phase 1: Train Teacher (PPO)

```
  Collect Trajectories → Compute GAE → PPO Loss → Update Teacher
```

Phase 2: Distill to Student

```
  Collect Teacher Trajectories → Distillation Loss → Update Student
```

**Characteristics:**

- ✓ Inductive bias from expert teacher
- ✓ Better sample efficiency than V1
- ✓ Faster convergence
- ~ Moderate speed
- ✗ Requires training teacher first

---

## 4. V3: Full System (LSTM + LLM)

```
┌────────────────────────────────────────────────────────────────┐
│      V3: Full System (LSTM + LLM Guided Strategy)              │
└────────────────────────────────────────────────────────────────┘

   Environment
       │
       ├─► Partial State (POMDP)      Full State (Complete)
       │   [qpos only, 8-dim]         [qpos + qvel, 17-dim]
       │                              │
       │                              │
       │                              ▼
       │                         ┌──────────────────┐
       │                         │ State            │
       │                         │ ▼                │
       │                         │ LLM Prompt       │
       │                         │ "Current state:..│
       │                         │  Predict action" │
       │                         │ │                │
       │                         │ ▼                │
       │                    ┌────────────────────┐ │
       │                    │ Ollama (Gemma3)    │ │
       │                    │ Generate Response  │ │
       │                    │ e.g. "Accelerate" │ │
       │                    └────┬───────────────┘ │
       │                         │                │
       │                         ▼                │
       │                    ┌──────────────────┐ │
       │                    │ Text-to-Embedding│ │
       │                    │ (256-dim z)      │ │
       │                    └────┬──────────────┘ │
       │                         │                │
       │                         ▼                │
       ▼                    ┌──────────────────┐  │
   ┌─────────────────────┐  │ Full State + z   │ │
   │ Student LSTM        │  │ [17 + 256 = 273] │ │
   │ (Learns)            │  │                  │ │
   │                     │  │ ▼                │
   │                     │  │┌──────────────┐  │ │
   │                     │  ││ Feat Encoder │  │ │
   │                     │  ││ (273→128)    │  │ │
   │                     │  │└────┬─────────┘  │ │
   │                     │  │     │            │ │
   │                     │  │┌────▼─────────┐  │ │
   │                     │  ││ LSTM         │  │ │
   │                     │  ││ hidden=128   │  │ │
   │                     │  │└────┬─────────┘  │ │
   │                     │  │     │            │ │
   │                     │  │┌────▼─────────┐  │ │
   │                     │  ││ Action Dist  │  │ │
   │                     │  ││ Head         │  │ │
   │                     │  │└────┬─────────┘  │ │
   │                     │  │     │            │ │
   │                     │  └─────┼────────────┘ │
   │                     │        │              │
   │  dist              │  teacher_dist
   │  hidden            │  teacher_hidden
   │  ▼                 │  ◄─────────────────────┘
   │┌──────────────────────────────────┐
   ││  DISTILLATION LOSS (V2 + LLM)    │
   ││                                  │
   ││  L = KL(S_dist || T_dist) +      │
   ││      MSE(S_hidden, T_hidden)     │
   │└──────────────────────────────────┘
   │  │
   │  ▼
   │┌──────────────────────────────────┐
   ││ Backprop to Student Only         │
   │└──────────────────────────────────┘
```

**Key Difference from V2:**

- Teacher receives `[full_state, llm_embedding]`
- LLM provides high-level strategy guidance
- Student still learns from partial observations

**Three-Phase Training:**

Phase 1: Train Teacher+LLM (PPO)

```
  Collect Traj → Get LLM guidance → Compute GAE → PPO Loss → Update Teacher
```

Phase 2: Distill to Student

```
  Collect Teacher+LLM Traj → Distillation Loss → Update Student
```

Phase 3: Evaluate

```
  Run Student with LLM guidance → Collect Metrics
```

**Characteristics:**

- ✓ Best performance (highest reward)
- ✓ Highest success rate
- ✓ LLM provides semantic understanding
- ✗ Slower inference (LLM API calls)
- ✗ Requires Ollama setup
- ~ More complex pipeline

---

## 5. Loss Functions

### V1: Policy Gradient Loss

```
L = E[-log π(a|s) + entropy_coeff × H(π)]

Where:
  π(a|s)  = Policy distribution (Normal)
  H(π)    = Entropy (encourages exploration)
```

### V2 & V3: Distillation Loss

```
L = α × KL(π_student || π_teacher) + β × MSE(h_student, h_teacher)

Where:
  KL()     = Kullback-Leibler divergence (action distributions)
  MSE()    = Mean Squared Error (hidden state alignment)
  α, β     = Loss weights (default: 1.0, 0.5)
  h        = LSTM hidden state outputs
```

---

## 6. Data Flow: Episode Generation

```
┌──────────────────────────────────────────────────────────────┐
│  Single Episode Training (Variant Example: V2)               │
└──────────────────────────────────────────────────────────────┘

Step 1: Initialization
┌────────────────────────────────┐
│ Reset Env: obs = [x,y,z,vx,vy]│
└────────────────┬───────────────┘
                 │
        Step-by-step loop:
                 │
Step 2: Get observation
┌────────────────▼───────────────┐
│ Student obs:   [x,y,z] (8-dim) │
│ Teacher state: [x,y,z,vx,vy]   │
│               (17-dim)          │
└────────────────┬───────────────┘
                 │
Step 3: Forward Pass
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼──────┐    ┌────▼──────────┐
    │ Student   │    │ Teacher       │
    │ Forward   │    │ Forward       │
    │ (s_out)   │    │ (t_out)       │
    └────┬──────┘    └────┬──────────┘
         │                │
Step 4: Sample Action
         │                │
    ┌────▼────────────────▼───┐
    │ a ~ π_student(s_out)    │
    │                         │
    │ (Teacher action         │
    │  stored for distill)    │
    └────┬───────────────────┘
         │
Step 5: Execute Action
         │
    ┌────▼────────────────────┐
    │ a' = env.step(a)        │
    │ r, done, obs' ◄─────┐   │
    └────┬────────────────┘   │
         │                    │
Step 6: Store Transition
         │
    ┌────▼───────────────────────┐
    │ Buffer.add(                 │
    │   obs, action, reward, done │
    │ )                           │
    └────┬───────────────────────┘
         │
Step 7: Check Episode Done
         │
    ├─Yes─► Finalize episode
    │        Compute return
    │
    └─No──► Next step
            (Step 2)
```

---

## 7. Training Loop Summary

```
┌─────────────────────────────────────────────────┐
│  Main Training Pipeline                         │
└─────────────────────────────────────────────────┘

Initialize Models
    │
    ▼
┌─────────────────────────────────────┐
│ For iteration = 1 to NUM_ITERATIONS │
└─────────────────────────────────────┘
    │
    ├──► Phase 1: Collect Data
    │    ├─► Run policy on env
    │    ├─► Store trajectories
    │    └─► Compute returns/advantages
    │
    ├──► Phase 2: Train Update
    │    ├─► Sample batch from buffer
    │    ├─► Forward pass
    │    ├─► Compute loss
    │    ├─► Backward pass
    │    └─► Optimizer step
    │
    ├──► Phase 3: Log Metrics
    │    ├─► Episode reward
    │    ├─► Success rate
    │    ├─► Model loss
    │    └─► Inference time
    │
    └──► Next iteration
         │
         ▼
    ┌──────────────────┐
    │ Evaluation Phase │
    └──────────────────┘
         │
         ├─► Run evaluation episodes
         ├─► Collect final metrics
         └─► Save model
             │
             ▼
    ┌──────────────────┐
    │ Generate Report  │
    │ & Visualizations │
    └──────────────────┘
```

---

## 8. Metrics Collection Pipeline

```
┌──────────────────────────────────────────────┐
│ Metrics Collection During Training           │
└──────────────────────────────────────────────┘

Episode Level:
  ├─► total_reward (sum of rewards)
  ├─► episode_length (num steps)
  ├─► success (reward > threshold)
  └─► avg_action_magnitude

Step Level:
  ├─► action (continuous vector)
  ├─► log_prob (from distribution)
  ├─► inference_time (ms)
  └─► memory_usage (MB)

Training Level:
  ├─► loss (KL + MSE or PG)
  ├─► gradient_norm
  ├─► learning_rate
  └─► convergence_metrics

Aggregation:
  │
  ▼
Variant Summary:
  ├─► mean_reward ± std
  ├─► success_rate (%)
  ├─► best_reward
  ├─► worst_reward
  └─► model_size (MB)

Comparison Across Variants:
  │
  ▼
Final Report (Markdown):
  ├─► Summary table
  ├─► Detailed metrics
  ├─► Analysis & findings
  └─► Conclusion
```

---

## 9. Environment Observation Structure

```
HalfCheetah-v4 Full Observation (17-dim):
┌─────────────────────────────────────┐
│ qpos (8-dim): Joint Positions       │
│ ├─► x: Torso x-position             │
│ ├─► y: Torso y-position             │
│ ├─► z: Torso z-position             │
│ ├─► θ0: Joint 0 angle               │
│ ├─► θ1: Joint 1 angle               │
│ ├─► θ2: Joint 2 angle               │
│ ├─► θ3: Joint 3 angle               │
│ └─► θ4: Joint 4 angle               │
│                                     │
│ qvel (9-dim): Joint Velocities      │
│ ├─► ẋ: Torso x-velocity             │
│ ├─► ẏ: Torso y-velocity             │
│ ├─► ż: Torso z-velocity             │
│ ├─► ω0: Joint 0 angular vel         │
│ ├─► ω1: Joint 1 angular vel         │
│ ├─► ω2: Joint 2 angular vel         │
│ ├─► ω3: Joint 3 angular vel         │
│ ├─► ω4: Joint 4 angular vel         │
│ └─► ω5: Foot contact velocity       │
└─────────────────────────────────────┘

POMDP Observation (8-dim):
└─► qpos only [x, y, z, θ0, θ1, θ2, θ3, θ4]
    (hides qvel)

Action Space (6-dim):
└─► Joint torques [τ0, τ1, τ2, τ3, τ4, τ5]

Reward:
└─► Forward velocity (x-position change)
    R = Δx - 0.1 × |action|²
```

---

## 10. Model Parameter Counts

```
Student LSTM:
  ├─ Feat Encoder:    8×64 + 64        = 576
  ├─ LSTM:            (64+128)×128×4   = 98,304
  ├─ Mu Head:         128×6 + 6        = 774
  ├─ Log Std:         1×6              = 6
  └─ Total:                              ~100K parameters

Teacher LSTM:
  ├─ Feat Encoder:    17×128 + 128     = 2,304
  ├─ LSTM:            (128+128)×128×4  = 131,072
  ├─ Mu Head:         128×6 + 6        = 774
  ├─ Log Std:         1×6              = 6
  └─ Total:                              ~134K parameters

Teacher LLM:
  ├─ Input:           (17 + 256) = 273
  ├─ Feat Encoder:    273×128 + 128×128 + 128 ≈ 52K
  ├─ LSTM:            (128+128)×128×4  = 131,072
  ├─ Mu Head:         128×6 + 6        = 774
  ├─ Log Std:         1×6              = 6
  └─ Total:                              ~184K parameters
```

---

This provides a complete visual understanding of the project architecture and data flow.
