#!/usr/bin/env python3
"""
Build script for the Rust osu_fast extension
Thanks claude
"""

import subprocess
import sys
import os

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Return code: {e.returncode}")
        return False

def main():
    print("Building Rust osu_rust extension...")
    
    # Check if we're in the osu_fast directory or need to find it
    if os.path.exists("Cargo.toml"):
        # We're already in the Rust project directory
        cwd = "."
    elif os.path.exists("osu_rust/Cargo.toml"):
        # We're in the parent directory
        cwd = "osu_rust"
    else:
        print("Error: Could not find Rust project. Make sure Cargo.toml exists.")
        sys.exit(1)
    
    # Build in release mode for maximum performance
    print("Compiling Rust extension (this may take a few minutes)...")
    success = run_command("maturin develop --release", cwd=cwd)
    
    if success:
        print("✅ Rust extension built successfully!")
        print("You can now use the accelerated functions in dataset.py")
        
        # Test import
        try:
            import osu_fast
            print("✅ Import test successful!")
        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            print("You may need to install maturin: pip install maturin")
    else:
        print("❌ Build failed!")
        print("Make sure you have:")
        print("  - Rust installed (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)")
        print("  - maturin installed (pip install maturin)")

if __name__ == "__main__":
    main()