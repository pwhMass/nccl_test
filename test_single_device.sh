#!/bin/bash
# Test with single device to avoid segfault

set -e

echo "=== Testing with single device fix ==="

# Set environment to use only bluerdma0
export LD_LIBRARY_PATH=/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/target/debug:/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib:/usr/local/cuda-13.0/lib64
export RUST_LOG=debug
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=bluerdma0  # Pin to bluerdma0 only
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# Build test
make clean
make MOCK=1

echo ""
echo "=== Running single device test ==="
mpirun -np 1 ./nccl_test 2>&1 | tee single_device_test.log

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Single device test successful!"
    echo ""
    echo "Now trying two processes with pinned device..."
    mpirun -np 2 ./nccl_test 2>&1 | tee two_process_test.log
else
    echo ""
    echo "✗ Single device test failed"
    echo ""
    echo "Let's try a different approach..."
fi