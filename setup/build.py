#!/usr/bin/env python3
"""
Script de build pour watermark-evil
Compile le code Rust et installe le package Python
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """Execute une commande et affiche sa sortie"""
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    result.check_returncode()
    return result

def main():
    # Chemin du projet
    project_root = Path(__file__).parent.parent.absolute()
    
    # Vérifier que maturin est installé
    try:
        import maturin
    except ImportError:
        print("Installing maturin...")
        run_command([sys.executable, "-m", "pip", "install", "maturin>=1.4,<2.0"])
    
    # Compiler le code Rust
    print("\nBuilding Rust code...")
    run_command(
        ["maturin", "develop", "--release"],
        cwd=project_root
    )
    
    # Installer les dépendances Python
    print("\nInstalling Python dependencies...")
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=project_root
    )
    
    print("\nBuild completed successfully!")

if __name__ == "__main__":
    main()
