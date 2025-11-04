#!/bin/bash
# NCCL Test Runner for Blue RDMA Driver (Mock Mode)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLUE_RDMA_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NCCL + Blue RDMA Driver Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse command line arguments
NUM_PROCS=${1:-2}
TEST_SIZE=${2:-1024}
MOCK_MODE=${MOCK:-1}

echo -e "${GREEN}Configuration:${NC}"
echo "  Number of processes: $NUM_PROCS"
echo "  Test data size: $TEST_SIZE"
echo "  Mock mode: $MOCK_MODE"
echo ""

# Set up environment
echo -e "${YELLOW}Setting up environment...${NC}"

BLUE_RDMA_IBVERBS="$BLUE_RDMA_ROOT/blue-rdma-driver/dtld-ibverbs"
RDMA_CORE_BUILD="$BLUE_RDMA_IBVERBS/rdma-core-55.0/build"

# Auto-detect CUDA installation
if [ -z "$CUDA_HOME" ]; then
    # Try common CUDA locations
    if [ -d "/usr/local/cuda-13.0" ]; then
        export CUDA_HOME="/usr/local/cuda-13.0"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.0" ]; then
        export CUDA_HOME="/usr/local/cuda-12.0"
    fi
fi

# Add CUDA to PATH if found
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/bin" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo -e "${GREEN}✓ CUDA found at: $CUDA_HOME${NC}"
fi

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA (nvcc) not found in PATH.${NC}"
    echo -e "${YELLOW}Hint: Set CUDA_HOME environment variable or add CUDA to PATH${NC}"
    echo -e "${YELLOW}Example: export PATH=/usr/local/cuda-13.0/bin:\$PATH${NC}"
    exit 1
fi

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    echo -e "${RED}Error: MPI (mpirun) not found. Please install OpenMPI or MPICH.${NC}"
    exit 1
fi

# Set library paths
export LD_LIBRARY_PATH="$BLUE_RDMA_IBVERBS/target/debug:$RDMA_CORE_BUILD/lib:$LD_LIBRARY_PATH"

# Set NCCL environment variables for IB transport
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=bluerdma
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL

# Set Blue RDMA driver debug level
export RUST_LOG=debug

echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  RUST_LOG: $RUST_LOG"
echo "  NCCL_NET: $NCCL_NET"
echo "  NCCL_IB_HCA: $NCCL_IB_HCA"
echo ""

# Build if necessary
if [ ! -f "$SCRIPT_DIR/nccl_test" ]; then
    echo -e "${YELLOW}Building test program...${NC}"
    cd "$SCRIPT_DIR"
    make MOCK=$MOCK_MODE
    echo ""
fi

# Check if Blue RDMA driver is built
if [ ! -f "$BLUE_RDMA_IBVERBS/target/debug/libbluerdma_rust.so" ]; then
    echo -e "${YELLOW}Building Blue RDMA driver...${NC}"
    cd "$BLUE_RDMA_IBVERBS"
    if [ "$MOCK_MODE" = "1" ]; then
        cargo build --no-default-features --features mock
    else
        cargo build
    fi
    echo ""
fi

# Run the test
echo -e "${GREEN}Running NCCL test...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"

cd "$SCRIPT_DIR"

# Run with MPI
if mpirun -np "$NUM_PROCS" --allow-run-as-root ./nccl_test 2>&1 | tee test_output.log; then
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${GREEN}✓ Test completed successfully!${NC}"

    # Check for errors in output
    if grep -q "ERROR" test_output.log || grep -q "FAILED" test_output.log; then
        echo -e "${RED}⚠ Warning: Errors detected in test output${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${RED}✗ Test failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Test Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"

# Extract and display key results
echo ""
echo "Checking test output..."
if grep -q "Test PASSED" test_output.log; then
    echo -e "${GREEN}✓ All ranks reported success${NC}"
else
    echo -e "${RED}✗ Some ranks failed${NC}"
fi

echo ""
echo "Log file saved to: test_output.log"
echo ""
