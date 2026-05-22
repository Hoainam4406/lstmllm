# Docker Deployment Guide - LSTM + LLM Ablation Study

## Quick Start

### Windows

```bash
# Double-click:
build_docker.bat

# Or from PowerShell:
docker build -t lstm-llm:latest .
docker run -it --rm -v "${PWD}\results:/app/results" -p 11434:11434 lstm-llm:latest
```

### Linux/Mac

```bash
# Run:
bash build_docker.sh

# Or manually:
docker build -t lstm-llm:latest .
docker run -it --rm -v $(pwd)/results:/app/results -p 11434:11434 lstm-llm:latest
```

---

## What's Included

### Dockerfile

- **Base**: Python 3.10-slim (optimized for size)
- **System deps**: MuJoCo, OpenGL, build tools
- **Python**: PyTorch, Gymnasium, Ollama integration
- **Model**: Auto-downloads gemma:4b (~2GB)
- **Pipeline**: Runs full ablation study (V1+V2+V3)

### docker-compose.yml

- **Ollama service**: Pre-configured on port 11434
- **App service**: Main pipeline with volume mounts
- **Network**: Internal communication
- **Health checks**: Ensures services are healthy

---

## Usage

### 1. Build Image

```bash
# Full build (first time, ~5-10 min)
docker build -t lstm-llm:latest .

# Rebuild after code changes
docker build -t lstm-llm:latest . --no-cache
```

### 2. Run Container

#### Interactive Mode (Best for testing)

```bash
docker run -it --rm \
  -v $(pwd)/results:/app/results \
  -p 11434:11434 \
  lstm-llm:latest bash

# Then inside container:
python ablation_pipeline.py --variants all
```

#### Detached Mode (For background running)

```bash
docker run -d \
  -v $(pwd)/results:/app/results \
  -p 11434:11434 \
  --name lstm_experiment \
  lstm-llm:latest

# Monitor logs:
docker logs -f lstm_experiment

# Stop when done:
docker stop lstm_experiment
```

#### Using docker-compose (Recommended)

```bash
# Start services
docker-compose up

# In another terminal, run pipeline:
docker-compose exec lstm_llm python ablation_pipeline.py --variants all

# Stop all services
docker-compose down
```

---

## Execution Flow

1. **Ollama Startup** (~10-15 sec)
   - Start Ollama server on port 11434
   - Verify connection

2. **Model Download** (~5-10 min on first run)
   - Downloads gemma:4b (~2GB)
   - Caches in /root/.ollama

3. **Pipeline Execution** (~20-30 min)
   - V1: LSTM only baseline
   - V2: LSTM + Teacher distillation
   - V3: Full system with LLM guidance

4. **Results Generation** (~1-2 min)
   - Markdown report with comparison table
   - PNG plots (reward, success rate, inference time)
   - Saved to `/app/results/ablation_study/`

---

## Volume Mounting

### Local Results

```bash
# Mount results directory to host
-v $(pwd)/results:/app/results

# Results available at:
# Host: ./results/ablation_study/
# Container: /app/results/ablation_study/
```

### Data Persistence

```bash
# Save Ollama models across runs
docker run -v ollama_models:/root/.ollama lstm-llm:latest

# Create named volume:
docker volume create ollama_models
```

---

## Memory & Performance

### Resource Requirements

```
RAM: 8GB minimum (4GB app + 4GB Ollama model)
CPU: 2 cores minimum
Disk: 10GB (for model + results)
```

### Optimization Tips

```bash
# Limit memory usage
docker run -m 8g lstm-llm:latest

# Set CPU limits
docker run --cpus="2" lstm-llm:latest

# Use mock LLM if OOM
docker run -e MOCK_LLM=1 lstm-llm:latest
```

---

## Troubleshooting

### Problem: Out of Memory

```bash
# Solution: Use smaller model
docker run -e OLLAMA_MODEL=phi lstm-llm:latest

# Or use mock interface:
docker run -e MOCK_LLM=1 lstm-llm:latest
```

### Problem: Port Already in Use

```bash
# Change Ollama port
docker run -p 11435:11434 lstm-llm:latest

# Or stop conflicting container:
docker ps
docker stop <container_id>
```

### Problem: Model Download Timeout

```bash
# Increase timeout
docker run --timeout 600 lstm-llm:latest

# Or pre-download model:
docker run ollama/ollama ollama pull gemma:4b
```

### Problem: Permission Denied on Results

```bash
# Fix ownership after container runs
sudo chown -R $USER:$USER results/

# Or run with your user ID:
docker run --user $(id -u):$(id -g) lstm-llm:latest
```

---

## Kaggle Deployment

### Option 1: Upload Docker Image

```bash
# Save image as tarball
docker save lstm-llm:latest | gzip > lstm-llm.tar.gz

# Upload to Kaggle Dataset
kaggle datasets create -p . --public

# In Kaggle notebook:
!docker load < lstm-llm.tar.gz
!docker run -p 11434:11434 lstm-llm:latest
```

### Option 2: Use Kaggle Kernel Directly

```python
# Install dependencies
!pip install -q gymnasium[mujoco] torch requests psutil

# Run mock (no Ollama)
!python ablation_pipeline.py --variants all
```

---

## Image Size & Optimization

```
Python 3.10-slim:      ~150 MB
PyTorch CPU:           ~500 MB
Gymnasium + MuJoCo:    ~300 MB
System dependencies:   ~200 MB
Ollama binary:         ~50 MB
Model (gemma:4b):      ~2.5 GB (runtime only)
────────────────────────────
Total base image:      ~1.2 GB
With model loaded:     ~3.7 GB in memory
```

---

## Advanced Usage

### Run Specific Variants Only

```bash
docker run lstm-llm:latest python ablation_pipeline.py --variants V1 V2
docker run lstm-llm:latest python ablation_pipeline.py --variants V3
```

### Run Demo (Quick Test)

```bash
docker run lstm-llm:latest python demo.py
```

### Custom Configuration

```bash
# Mount custom config
docker run -v $(pwd)/config.py:/app/config.py lstm-llm:latest

# Set environment variables
docker run -e NUM_EPISODES=20 lstm-llm:latest
```

### Interactive Shell

```bash
# Get bash access
docker run -it lstm-llm:latest bash

# Inside container:
ollama serve &
sleep 5
python ablation_pipeline.py --variants all
```

---

## Best Practices

✅ **DO:**

- Use docker-compose for complex setups
- Mount results directory for persistence
- Test locally before uploading to Kaggle
- Use named volumes for model caching
- Check logs if something fails

❌ **DON'T:**

- Run multiple containers without resource limits
- Store results only in container (use volumes!)
- Upload without testing locally first
- Use latest tag in production
- Forget to commit code changes

---

## Cleanup

```bash
# Remove image
docker rmi lstm-llm:latest

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a
```

---

## Support

For issues:

1. Check Docker logs: `docker logs <container_id>`
2. Verify Ollama: `curl http://localhost:11434/api/tags`
3. Check resources: `docker stats`
4. Review README.md for pipeline details
