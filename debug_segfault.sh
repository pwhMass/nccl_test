#!/bin/bash
# Debug script for NCCL segmentation fault

set -e

echo "=== Building with debug symbols ==="
make clean
make MOCK=1 CXXFLAGS="-std=c++11 -O0 -g -Wall"

echo ""
echo "=== Running with GDB to capture segfault ==="
cd /home/peng/projects/rdma_all/nccl_test

# Set environment
export LD_LIBRARY_PATH=/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/target/debug:/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib:/usr/local/cuda-13.0/lib64
export RUST_LOG=debug
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=bluerdma
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Create GDB script
cat > gdb_script.txt << 'EOF'
set pagination off
set confirm off
handle SIGSEGV nostop noprint pass
handle SIGPIPE nostop noprint pass
run
bt
info registers
thread apply all bt
quit
EOF

# Run with GDB
echo "Starting GDB debug session..."
gdb -batch -x gdb_script.txt ./nccl_test 2>&1 | tee gdb_debug.log

echo ""
echo "=== Debug output saved to gdb_debug.log ==="
echo ""

# Extract key information
echo "=== Stack trace analysis ==="
grep -A 20 "Program received signal\|#0\|#1" gdb_debug.log | head -30