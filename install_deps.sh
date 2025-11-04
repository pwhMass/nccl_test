#!/bin/bash
# NCCL + MPI Installation Script for WSL/Ubuntu 24.04
# This script installs NCCL and OpenMPI for Blue RDMA driver testing

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NCCL + MPI Installation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect architecture and distro
ARCH=$(dpkg --print-architecture)
DISTRO="ubuntu2404"  # Ubuntu 24.04

echo -e "${GREEN}Detected Configuration:${NC}"
echo "  Architecture: $ARCH"
echo "  Distribution: $DISTRO"
echo ""

# Check CUDA installation
if [ -d "/usr/local/cuda-13.0" ]; then
    CUDA_VERSION="13.0"
    CUDA_HOME="/usr/local/cuda-13.0"
    echo -e "${GREEN}✓ Found CUDA $CUDA_VERSION at $CUDA_HOME${NC}"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    CUDA_VERSION=$(cat $CUDA_HOME/version.json | grep -oP '"cuda".*?"version".*?"\K[0-9.]+' | head -1)
    echo -e "${GREEN}✓ Found CUDA $CUDA_VERSION at $CUDA_HOME${NC}"
else
    echo -e "${RED}✗ CUDA not found. Please install CUDA first.${NC}"
    exit 1
fi
echo ""

# Step 1: Install OpenMPI
echo -e "${YELLOW}Step 1: Installing OpenMPI...${NC}"
if command -v mpirun &> /dev/null; then
    echo -e "${GREEN}✓ OpenMPI already installed${NC}"
    mpirun --version | head -1
else
    echo "Installing OpenMPI via apt..."
    sudo apt update
    sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
    echo -e "${GREEN}✓ OpenMPI installed successfully${NC}"
fi
echo ""

# Step 2: Install NCCL via Network Repository
echo -e "${YELLOW}Step 2: Installing NCCL via Network Repository...${NC}"

# Check if NCCL is already installed
if dpkg -l | grep -q libnccl2; then
    INSTALLED_VERSION=$(dpkg -l | grep libnccl2 | awk '{print $3}')
    echo -e "${GREEN}✓ NCCL already installed (version: $INSTALLED_VERSION)${NC}"

    read -p "Do you want to reinstall/upgrade NCCL? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping NCCL installation."
        echo ""
        echo -e "${GREEN}Installation completed!${NC}"
        exit 0
    fi
fi

# Download and install CUDA keyring
echo "Adding NVIDIA CUDA repository..."
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"

if [ -f "cuda-keyring_1.1-1_all.deb" ]; then
    echo "Keyring package already downloaded."
else
    echo "Downloading CUDA keyring from: $KEYRING_URL"
    wget -q --show-progress "$KEYRING_URL" || {
        echo -e "${RED}Failed to download keyring. Trying alternative URL...${NC}"
        # Fallback to older Ubuntu version if 24.04 not available
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${ARCH}/cuda-keyring_1.1-1_all.deb"
        wget -q --show-progress "$KEYRING_URL"
    }
fi

sudo dpkg -i cuda-keyring_1.1-1_all.deb
echo -e "${GREEN}✓ CUDA repository added${NC}"
echo ""

# Update APT database
echo "Updating APT database..."
sudo apt update

# Install NCCL with version locking to avoid CUDA upgrade
echo "Installing NCCL packages..."
echo -e "${YELLOW}Note: Installing NCCL without upgrading CUDA${NC}"

# Find compatible NCCL version for CUDA 13.0
echo "Searching for NCCL packages compatible with CUDA 13.0..."
sudo apt-cache search libnccl2 | grep -E "libnccl2.*cuda" || echo "No CUDA-specific packages found"

# Install NCCL (this will use the latest compatible version)
# We'll install without specifying CUDA version, as apt will handle dependencies
sudo apt install -y libnccl2 libnccl-dev

# Prevent automatic CUDA upgrade
echo "Marking CUDA packages to prevent automatic upgrade..."
if dpkg -l | grep -q "cuda-toolkit-13-0"; then
    sudo apt-mark hold cuda-toolkit-13-0
fi

echo -e "${GREEN}✓ NCCL installed successfully${NC}"
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"

# Check MPI
if command -v mpirun &> /dev/null; then
    echo -e "${GREEN}✓ MPI:${NC} $(mpirun --version | head -1)"
else
    echo -e "${RED}✗ MPI not found${NC}"
fi

# Check NCCL
if [ -f "/usr/include/nccl.h" ]; then
    NCCL_VERSION=$(grep "NCCL_VERSION_CODE" /usr/include/nccl.h | awk '{print $3}')
    NCCL_MAJOR=$((NCCL_VERSION / 10000))
    NCCL_MINOR=$(((NCCL_VERSION / 100) % 100))
    NCCL_PATCH=$((NCCL_VERSION % 100))
    echo -e "${GREEN}✓ NCCL:${NC} v${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}"
else
    echo -e "${RED}✗ NCCL headers not found${NC}"
fi

# Check CUDA
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo -e "${GREEN}✓ CUDA:${NC} $($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
else
    echo -e "${RED}✗ CUDA compiler not found${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Installation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Add CUDA to your PATH:"
echo "   export PATH=$CUDA_HOME/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "2. Build the NCCL test program:"
echo "   cd $(dirname "$0")"
echo "   make MOCK=1"
echo ""
echo "3. Run the test:"
echo "   ./run_test.sh"
echo ""
echo -e "${GREEN}Installation completed!${NC}"
