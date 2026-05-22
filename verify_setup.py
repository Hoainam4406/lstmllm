#!/usr/bin/env python3
"""
Final Verification Checklist for LSTM + LLM Ablation Study Pipeline
Run this after creating everything to ensure all components are in place
"""

import os
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def check_file(path, required=True):
    """Check if file exists"""
    exists = Path(path).exists()
    status = f"{Colors.GREEN}✓{Colors.RESET}" if exists else f"{Colors.RED}✗{Colors.RESET}"
    req = "REQUIRED" if required else "optional"
    print(f"  {status} {path:50s} ({req})")
    return exists


def main():
    print("\n" + "="*80)
    print(f"{Colors.BLUE}LSTM + LLM ABLATION STUDY - FINAL VERIFICATION CHECKLIST{Colors.RESET}")
    print("="*80 + "\n")
    
    # Core pipeline files
    print(f"{Colors.YELLOW}CORE PIPELINE FILES (Required){Colors.RESET}")
    print("-" * 80)
    
    core_files = [
        "ablation_pipeline.py",
        "config.py",
        "models.py",
        "utils.py",
        "llm_interface.py",
        "training.py",
        "distillation.py",
        "evaluation.py",
        "PO-MuJoCo.py",
    ]
    
    core_ok = all(check_file(f, required=True) for f in core_files)
    print()
    
    # Documentation files
    print(f"{Colors.YELLOW}DOCUMENTATION FILES (Recommended){Colors.RESET}")
    print("-" * 80)
    
    doc_files = [
        "README.md",
        "USAGE_GUIDE.md",
        "PROJECT_SUMMARY.md",
        "ARCHITECTURE.md",
        "QUICKSTART.txt",
    ]
    
    doc_ok = all(check_file(f, required=False) for f in doc_files)
    print()
    
    # Utility scripts
    print(f"{Colors.YELLOW}UTILITY SCRIPTS (Recommended){Colors.RESET}")
    print("-" * 80)
    
    util_files = [
        "setup_check.py",
        "demo.py",
        "quickstart.bat",
    ]
    
    util_ok = all(check_file(f, required=False) for f in util_files)
    print()
    
    # Configuration
    print(f"{Colors.YELLOW}CONFIGURATION FILES (Required){Colors.RESET}")
    print("-" * 80)
    
    config_files = [
        "requirements.txt",
    ]
    
    config_ok = all(check_file(f, required=True) for f in config_files)
    print()
    
    # Check old files still present
    print(f"{Colors.YELLOW}LEGACY FILES (Optional - can be removed){Colors.RESET}")
    print("-" * 80)
    
    legacy_files = [
        "Asymmetric.py",
        "Loss.py",
    ]
    
    for f in legacy_files:
        status = "✓ Found" if Path(f).exists() else "✗ Not found"
        print(f"  {status:10s} {f:50s} (can delete if replaced)")
    print()
    
    # Summary
    print("="*80)
    print(f"{Colors.YELLOW}VERIFICATION SUMMARY{Colors.RESET}")
    print("="*80)
    
    print(f"\nCore Files:          {Colors.GREEN}✓ PASS{Colors.RESET}" if core_ok else f"\nCore Files:          {Colors.RED}✗ FAIL{Colors.RESET}")
    print(f"Documentation:       {Colors.GREEN}✓ PASS{Colors.RESET}" if doc_ok else f"Documentation:       {Colors.YELLOW}⚠ INCOMPLETE{Colors.RESET}")
    print(f"Utilities:           {Colors.GREEN}✓ PASS{Colors.RESET}" if util_ok else f"Utilities:           {Colors.YELLOW}⚠ INCOMPLETE{Colors.RESET}")
    print(f"Configuration:       {Colors.GREEN}✓ PASS{Colors.RESET}" if config_ok else f"Configuration:       {Colors.RED}✗ FAIL{Colors.RESET}")
    
    print("\n" + "="*80)
    
    if core_ok and config_ok:
        print(f"{Colors.GREEN}✓ PROJECT READY TO RUN!{Colors.RESET}\n")
        print("Next steps:")
        print("  1. python setup_check.py      # Verify dependencies")
        print("  2. python demo.py             # Quick test (2-3 min)")
        print("  3. python ablation_pipeline.py --variants all  # Full run")
        print()
        return 0
    else:
        print(f"{Colors.RED}✗ PROJECT INCOMPLETE - Please fix missing files above{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
