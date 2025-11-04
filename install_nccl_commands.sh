#!/bin/bash
# NCCL Installation Commands for WSL Ubuntu 24.04 + CUDA 13.0
# Run this script or execute commands one by one

set -e

echo "========================================="
echo "  NCCL Installation for Ubuntu 24.04"
echo "  CUDA 13.0 Compatible"
echo "========================================="
echo ""

# Step 1: Install OpenMPI
echo "Step 1: Installing OpenMPI..."
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
echo ""

# Step 2: Check available NCCL versions from Ubuntu repository
echo "Step 2: Checking available NCCL versions..."
apt-cache policy libnccl2
echo ""

# Step 3: Install NCCL from Ubuntu repository (version 2.18.5)
# This is the stable version available for Ubuntu 24.04
echo "Step 3: Installing NCCL from Ubuntu repository..."
echo "Note: Installing NCCL 2.18.5 (Ubuntu 24.04 official version)"
echo "This version works with CUDA 13.0"
sudo apt install -y libnccl2 libnccl-dev
echo ""

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
echo ""

# Check MPI
if command -v mpirun &> /dev/null; then
    echo "✓ MPI: $(mpirun --version | head -1)"
else
    echo "✗ MPI not found"
fi

# Check NCCL
if [ -f "/usr/include/nccl.h" ]; then
    NCCL_VERSION=$(grep "NCCL_VERSION_CODE" /usr/include/nccl.h | awk '{print $3}')
    NCCL_MAJOR=$((NCCL_VERSION / 10000))
    NCCL_MINOR=$(((NCCL_VERSION / 100) % 100))
    NCCL_PATCH=$((NCCL_VERSION % 100))
    echo "✓ NCCL: v${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}"
    echo "  Header: /usr/include/nccl.h"
else
    echo "✗ NCCL headers not found"
fi

# Check NCCL library
if [ -f "/usr/lib/x86_64-linux-gnu/libnccl.so.2" ]; then
    echo "✓ NCCL library: /usr/lib/x86_64-linux-gnu/libnccl.so.2"
else
    echo "✗ NCCL library not found"
fi

# Check CUDA
if [ -f "/usr/local/cuda-13.0/bin/nvcc" ]; then
    CUDA_VER=$(/usr/local/cuda-13.0/bin/nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA: $CUDA_VER"
else
    echo "⚠ CUDA 13.0 not found"
fi

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "========================================="
echo ""
echo "IMPORTANT NOTES:"
echo "1. NCCL 2.18.5 is compatible with CUDA 13.0"
echo "2. This is the official Ubuntu 24.04 version"
echo "3. Newer NCCL versions (2.28.x) require CUDA 12.x"
echo ""
echo "Next steps:"
echo "  cd /home/peng/projects/rdma_all/nccl_test"
echo "  ./check_env.sh      # Verify environment"
echo "  make MOCK=1         # Build test"
echo "  ./run_test.sh       # Run test"
echo ""
