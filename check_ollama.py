#!/usr/bin/env python
"""Check Ollama availability and memory before running pipeline"""

import requests
import psutil
import json
from config import OLLAMA_HOST, OLLAMA_MODEL

print("="*70)
print("OLLAMA SYSTEM CHECK")
print("="*70)

# 1. Check Ollama connectivity
print("\n[1] Checking Ollama Connectivity...")
try:
    response = requests.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5)
    if response.status_code == 200:
        print("✓ Ollama is running")
        models = response.json().get("models", [])
        print(f"  Available models: {[m.get('name') for m in models]}")
    else:
        print(f"✗ Ollama returned status {response.status_code}")
        print("  Solution: Start Ollama with: ollama serve")
        print("            or: ollama run qwen2:7b")
except Exception as e:
    print(f"✗ Cannot connect to Ollama at {OLLAMA_HOST}")
    print(f"  Error: {e}")
    print("  Solution: Start Ollama with: ollama serve")
    print("            or: ollama run qwen2:7b")

# 2. Check system memory
print("\n[2] Checking System Memory...")
memory = psutil.virtual_memory()
available_gb = memory.available / (1024**3)
total_gb = memory.total / (1024**3)

print(f"  Total RAM: {total_gb:.1f} GiB")
print(f"  Available RAM: {available_gb:.1f} GiB")
print(f"  Used: {memory.percent}%")

# Model memory requirements
model_requirements = {
    "qwen2:7b": 2.4,
    "qwen2:4b": 1.8,
    "gemma3:4b": 1.5,
    "llama3:latest": 2.7,
    "mistral:latest": 2.0,
}

print(f"\n[3] Memory Requirements for Models:")
for model, needed_gb in model_requirements.items():
    status = "✓" if available_gb >= needed_gb else "✗"
    print(f"  {status} {model}: {needed_gb} GiB (available: {available_gb:.1f} GiB)")

# 3. Recommendation
print("\n[4] RECOMMENDATION:")
suitable_models = [m for m, gb in model_requirements.items() if available_gb >= gb]

if suitable_models:
    print(f"✓ Available models with sufficient memory:")
    for model in suitable_models:
        print(f"  - {model}")
    print(f"\nRun: ollama run {suitable_models[0]}")
else:
    print("✗ NO suitable models! All require more than available memory.")
    print(f"\nSolution:")
    print("  1. Close other applications to free up memory")
    print("  2. Increase virtual memory/pagefile")
    print("  3. Use mock LLM interface (will run V1/V2 only, mock V3)")
    print("\nTo use Mock LLM (no real Ollama needed):")
    print("  python ablation_pipeline.py --variants V1 V2")

# 4. Check if Ollama can run a prompt
print("\n[5] Testing Prompt Generation (this may take a moment)...")
try:
    # Try to generate with model
    response = requests.post(
        f"http://{OLLAMA_HOST}/api/generate",
        json={
            "model": "qwen2:7b",
            "prompt": "Say 'ok'",
            "stream": False
        },
        timeout=30
    )
    
    if response.status_code == 200:
        print("✓ Can generate prompts successfully!")
    else:
        error = response.json().get("error", "Unknown error")
        if "resource limitations" in error or "more system memory" in error:
            print(f"✗ {error}")
            print("  → Not enough memory to load model")
            print("  → Will use Mock LLM interface instead")
        else:
            print(f"✗ Error: {error}")
except requests.exceptions.Timeout:
    print("✗ Model generation timed out (likely loading)")
except Exception as e:
    print(f"✗ Cannot test prompt generation: {e}")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
if available_gb < 2.4:
    print(f"""
Available RAM ({available_gb:.1f} GiB) is below model requirements (2.4+ GiB)

OPTIONS:
  1. SKIP Ollama - Run V1+V2 only:
     python ablation_pipeline.py --variants V1 V2
  
  2. FREE UP MEMORY - Close apps, then try V1+V2+V3:
     (V3 will use mock interface if Ollama still unavailable)
     python ablation_pipeline.py --variants all
  
  3. UNLOAD MODELS - In Ollama:
     - Stop the running model (Ctrl+C in Ollama terminal)
     - Then rerun pipeline
""")
else:
    print(f"""
✓ System ready! Available memory: {available_gb:.1f} GiB

Run full pipeline:
  python ablation_pipeline.py --variants all
""")
