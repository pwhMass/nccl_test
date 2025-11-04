#!/bin/bash
# Test with NCCL environment fixes

set -e

echo "=== Testing with NCCL environment fixes ==="

# Set environment to limit device discovery
export LD_LIBRARY_PATH=/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/target/debug:/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib:/usr/local/cuda-13.0/lib64
export RUST_LOG=debug
export NCCL_IB_DISABLE=0
export NCCL_NET=IB

# Force NCCL to use only one device
export NCCL_IB_HCA=bluerdma0
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL's multi-device optimization
export NCCL_TREE_THRESHOLD=0
export NCCL_RING_THRESHOLD=0
export NCCL_DEBUG=WARN

# Try to disable second device
export NCCL_IB_RETRY_CNT=0
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22

# Build test
make clean
make MOCK=1

echo ""
echo "=== Testing single GPU, single device ==="
mpirun -np 1 ./nccl_test 2>&1 | tee test_single.log

# Check for success
if grep -q "Test PASSED" test_single.log; then
    echo ""
    echo "✓ Single GPU test successful!"

    echo ""
    echo "=== Testing two GPUs (GPU 0 only) ==="
    # Try with 2 processes but only GPU 0
    mpirun -np 2 ./nccl_test 2>&1 | tee test_two.log

    if grep -q "Test PASSED" test_two.log; then
        echo "✓ Two process test successful!"
    else
        echo "✗ Two process test failed"
        echo "This may be expected with single GPU"
    fi
else
    echo "✗ Single GPU test failed"
    echo "Checking for errors in log..."
    grep -E "ERROR|FAILED|Signal" test_single.log | head -5
fi