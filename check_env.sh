#!/bin/bash
# Environment Verification Script for NCCL + Blue RDMA Testing

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Environment Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track overall status
ALL_OK=true

# Function to check command
check_command() {
    local cmd=$1
    local name=$2
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $name: $(command -v $cmd)"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not found"
        ALL_OK=false
        return 1
    fi
}

# Function to check file
check_file() {
    local file=$1
    local name=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $name: $file"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not found at $file"
        ALL_OK=false
        return 1
    fi
}

# Function to check directory
check_dir() {
    local dir=$1
    local name=$2
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $name: $dir"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not found at $dir"
        ALL_OK=false
        return 1
    fi
}

echo -e "${YELLOW}Checking CUDA...${NC}"
# Check CUDA installation
if [ -d "/usr/local/cuda-13.0" ]; then
    CUDA_HOME="/usr/local/cuda-13.0"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
else
    echo -e "${RED}✗${NC} CUDA: Not found"
    ALL_OK=false
    CUDA_HOME=""
fi

if [ -n "$CUDA_HOME" ]; then
    check_dir "$CUDA_HOME" "CUDA Home"
    check_file "$CUDA_HOME/bin/nvcc" "nvcc compiler"
    check_file "$CUDA_HOME/lib64/libcudart.so" "CUDA runtime"

    # Show CUDA version
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
        echo -e "  Version: ${BLUE}$CUDA_VERSION${NC}"
    fi
fi
echo ""

echo -e "${YELLOW}Checking NCCL...${NC}"
# Check NCCL installation
if [ -f "/usr/include/nccl.h" ]; then
    check_file "/usr/include/nccl.h" "NCCL header"

    # Extract NCCL version
    NCCL_VERSION=$(grep "NCCL_VERSION_CODE" /usr/include/nccl.h | awk '{print $3}')
    NCCL_MAJOR=$((NCCL_VERSION / 10000))
    NCCL_MINOR=$(((NCCL_VERSION / 100) % 100))
    NCCL_PATCH=$((NCCL_VERSION % 100))
    echo -e "  Version: ${BLUE}${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}${NC}"

    # Check NCCL library
    if [ -f "/usr/lib/x86_64-linux-gnu/libnccl.so" ]; then
        check_file "/usr/lib/x86_64-linux-gnu/libnccl.so" "NCCL library"
    elif [ -f "/usr/local/nccl/lib/libnccl.so" ]; then
        check_file "/usr/local/nccl/lib/libnccl.so" "NCCL library"
    else
        echo -e "${RED}✗${NC} NCCL library: Not found"
        ALL_OK=false
    fi
else
    echo -e "${RED}✗${NC} NCCL: Not installed"
    echo -e "  ${YELLOW}Run: ./install_deps.sh to install${NC}"
    ALL_OK=false
fi
echo ""

echo -e "${YELLOW}Checking MPI...${NC}"
check_command "mpirun" "mpirun"
check_command "mpicxx" "mpicxx"
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>&1 | head -1)
    echo -e "  Version: ${BLUE}$MPI_VERSION${NC}"
fi
echo ""

echo -e "${YELLOW}Checking Blue RDMA Driver...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLUE_RDMA_ROOT="$(dirname "$SCRIPT_DIR")"
BLUE_RDMA_IBVERBS="$BLUE_RDMA_ROOT/dtld-ibverbs"
RDMA_CORE_BUILD="$BLUE_RDMA_IBVERBS/rdma-core-55.0/build"

check_dir "$BLUE_RDMA_IBVERBS" "dtld-ibverbs"
check_file "$BLUE_RDMA_IBVERBS/target/debug/libbluerdma_rust.so" "Blue RDMA Rust library"
check_dir "$RDMA_CORE_BUILD" "rdma-core build"
check_file "$RDMA_CORE_BUILD/lib/libibverbs.so.1" "libibverbs"
check_file "$RDMA_CORE_BUILD/lib/libbluerdma-rdmav34.so" "Blue RDMA provider"

# Check ibv_devices
if [ -f "$RDMA_CORE_BUILD/bin/ibv_devices" ]; then
    check_file "$RDMA_CORE_BUILD/bin/ibv_devices" "ibv_devices"
fi
echo ""

echo -e "${YELLOW}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} nvidia-smi: Available"
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
    echo -e "  GPU: ${BLUE}$GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi: Not found (GPU may not be available)"
    echo -e "  ${YELLOW}Note: This is OK for WSL without GPU support${NC}"
fi
echo ""

echo -e "${BLUE}========================================${NC}"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✓ All required components are installed!${NC}"
    echo ""
    echo -e "You can now build and run the test:"
    echo -e "  ${BLUE}cd $SCRIPT_DIR${NC}"
    echo -e "  ${BLUE}make info${NC}    # Show build configuration"
    echo -e "  ${BLUE}make MOCK=1${NC}  # Build the test"
    echo -e "  ${BLUE}./run_test.sh${NC}  # Run the test"
else
    echo -e "${RED}✗ Some components are missing${NC}"
    echo ""
    echo -e "To install missing components:"
    if ! command -v mpirun &> /dev/null || [ ! -f "/usr/include/nccl.h" ]; then
        echo -e "  ${BLUE}./install_deps.sh${NC}  # Install NCCL and MPI"
    fi
    if [ ! -f "$BLUE_RDMA_IBVERBS/target/debug/libbluerdma_rust.so" ]; then
        echo -e "  ${BLUE}cd $BLUE_RDMA_IBVERBS && cargo build --no-default-features --features mock${NC}"
    fi
    if [ ! -f "$RDMA_CORE_BUILD/lib/libibverbs.so.1" ]; then
        echo -e "  ${BLUE}cd $RDMA_CORE_BUILD/.. && ./build.sh${NC}"
    fi
fi
echo -e "${BLUE}========================================${NC}"
