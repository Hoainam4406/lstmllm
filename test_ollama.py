#!/usr/bin/env python
"""Quick test script to check Ollama connectivity"""

import requests
import json
from config import OLLAMA_HOST, OLLAMA_MODEL

print("="*60)
print("Testing Ollama Connection")
print("="*60)
print(f"\nOllama Host: {OLLAMA_HOST}")
print(f"Ollama Model: {OLLAMA_MODEL}")

# Test connection
try:
    print("\n[1] Testing basic connectivity...")
    response = requests.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5)
    print(f"✓ Connected to Ollama (Status: {response.status_code})")
    
    # Show available models
    models = response.json().get("models", [])
    print(f"\nAvailable models ({len(models)}):")
    for model in models:
        print(f"  - {model.get('name', 'unknown')}")
    
    # Check if our model exists
    model_names = [m.get('name', '') for m in models]
    if OLLAMA_MODEL in model_names or any(OLLAMA_MODEL in name for name in model_names):
        print(f"\n✓ Model '{OLLAMA_MODEL}' is available!")
    else:
        print(f"\n✗ Model '{OLLAMA_MODEL}' NOT found in available models")
    
    # Test prompt
    print("\n[2] Testing prompt generation...")
    prompt = "What is the optimal action for a robot in state [1.0, 0.5, 0.2]?"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    
    response = requests.post(
        f"http://{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        response_text = result.get('response', '').strip()
        print(f"✓ Prompt generation successful!")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response_text[:200]}...")
        print("\n✓ OLLAMA IS WORKING!")
    else:
        print(f"✗ Failed to generate prompt (Status: {response.status_code})")
        print(f"Response: {response.text}")

except requests.exceptions.ConnectionError as e:
    print(f"\n✗ Connection Error: Cannot connect to Ollama at {OLLAMA_HOST}")
    print(f"   Make sure Ollama is running: ollama serve")
    print(f"   Error: {e}")

except requests.exceptions.Timeout:
    print(f"\n✗ Timeout: Ollama took too long to respond")

except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n" + "="*60)
