#!/bin/bash

# MiniTorch Build Script
set -e

echo "🚀 Building MiniTorch..."

# Check if we have the required dependencies
echo "📋 Checking dependencies..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found. Please install CUDA Toolkit."
    exit 1
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake."
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv."
    exit 1
fi

# Get the Python executable that uv uses
PYTHON_EXECUTABLE=$(uv run python -c "import sys; print(sys.executable)")
echo "🐍 Using Python: $PYTHON_EXECUTABLE"

# Install required dependencies
echo "🔧 Setting up Python dependencies..."
echo "Installing build dependencies..."
uv pip install setuptools>=60 pybind11>=2.10.0 cmake

# Build and install the Python package (this will handle CMake automatically)
echo "🏗️ Building and installing Python package..."

# Install Python package
echo "📦 Installing Python package..."
uv pip install -e .

echo "🎉 MiniTorch is ready!"
echo ""
echo "💡 Try running the examples:"
echo "   uv run python examples/simple_example.py"
echo "   uv run python examples/neural_network.py" 