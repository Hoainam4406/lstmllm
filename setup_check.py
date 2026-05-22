#!/usr/bin/env python
"""
Quick Setup Checker for LSTM + LLM Ablation Study
Verifies all dependencies and environment configuration
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠ Warning: Python 3.9+ recommended")
        return False
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT FOUND")
        return False


def check_dependencies():
    """Check all required dependencies"""
    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60 + "\n")
    
    required_packages = [
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("requests", "requests"),
        ("psutil", "psutil"),
    ]
    
    all_ok = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ All dependencies installed!")
    else:
        print("✗ Some dependencies missing. Run: pip install -r requirements.txt")
    print("="*60 + "\n")
    
    return all_ok


def check_files():
    """Check required files exist"""
    print("\n" + "="*60)
    print("FILE CHECK")
    print("="*60 + "\n")
    
    required_files = [
        "config.py",
        "models.py",
        "utils.py",
        "llm_interface.py",
        "training.py",
        "distillation.py",
        "evaluation.py",
        "ablation_pipeline.py",
        "PO-MuJoCo.py",
        "requirements.txt",
        "README.md",
    ]
    
    all_exist = True
    for filename in required_files:
        path = Path(filename)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False
    
    print("\n" + "="*60)
    if all_exist:
        print("✓ All required files present!")
    else:
        print("✗ Some files missing")
    print("="*60 + "\n")
    
    return all_exist


def check_ollama():
    """Check if Ollama is available"""
    print("\n" + "="*60)
    print("OLLAMA CHECK (Optional for V3 variant)")
    print("="*60 + "\n")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running at http://localhost:11434")
            data = response.json()
            if data.get("models"):
                print(f"✓ Found {len(data['models'])} model(s):")
                for model in data["models"]:
                    print(f"  - {model.get('name', 'unknown')}")
            return True
        else:
            print("✗ Ollama server not responding properly")
            return False
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        print("\nTo enable V3 variant, run:")
        print("  ollama serve  (or use Docker)")
        print("  ollama pull gemma3")
        return False


def check_mujoco():
    """Check MuJoCo environment"""
    print("\n" + "="*60)
    print("MUJOCO ENVIRONMENT CHECK")
    print("="*60 + "\n")
    
    try:
        import gymnasium as gym
        from PO_MuJoCo import POMuJoCoWrapper
        
        print("Initializing HalfCheetah-v4...")
        env = gym.make("HalfCheetah-v4")
        po_env = POMuJoCoWrapper(env)
        
        obs, _ = po_env.reset()
        print(f"✓ HalfCheetah environment working")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation space: {po_env.observation_space}")
        print(f"  Action space: {po_env.action_space}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ MuJoCo environment error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check MuJoCo is installed: pip install gymnasium[mujoco]")
        print("  2. May need to install mujoco: pip install mujoco")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*70)
    print(" "*15 + "LSTM + LLM Ablation Study Setup Checker")
    print("="*70 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("MuJoCo Environment", check_mujoco),
        ("Ollama (Optional)", check_ollama),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n✗ Error during {check_name} check: {e}\n")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    critical = ["Python Version", "Dependencies", "Files", "MuJoCo Environment"]
    critical_ok = all(results.get(c, False) for c in critical)
    
    print("Critical Checks:")
    for check_name in critical:
        status = "✓ PASS" if results.get(check_name) else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    print("\nOptional Checks:")
    print(f"  {'✓' if results.get('Ollama (Optional)') else '✗'} Ollama (needed for V3 variant)")
    
    print("\n" + "="*70)
    if critical_ok:
        print("\n✓ Setup complete! You can now run the pipeline:")
        print("\n  python ablation_pipeline.py --variants all\n")
        if not results.get("Ollama (Optional)"):
            print("Note: V3 variant requires Ollama. Run V1 and V2 with:")
            print("  python ablation_pipeline.py --variants V1 V2\n")
    else:
        print("\n✗ Setup incomplete. Fix the errors above and try again.\n")
    
    print("="*70 + "\n")
    
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
