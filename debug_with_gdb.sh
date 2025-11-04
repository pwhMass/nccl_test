#!/bin/bash
# Better GDB debug script for NCCL segfault

set -e

echo "=== Building with debug symbols ==="
make clean
make MOCK=1 CXXFLAGS="-std=c++11 -O0 -g -Wall"

# Set environment
export LD_LIBRARY_PATH=/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/target/debug:/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib:/usr/local/cuda-13.0/lib64
export RUST_LOG=debug
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=bluerdma
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Create a simpler test program first
cat > simple_test.cpp << 'EOF'
#include <cstdio>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

int main(int argc, char *argv[])
{
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Rank %d/%d starting\n", rank, world_size);

    // Initialize NCCL with minimal setup
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
        printf("Rank 0 generated NCCL ID\n");
    }

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // This is where the crash happens
    printf("Rank %d about to call ncclCommInitRank\n", rank);
    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);
    printf("Rank %d ncclCommInitRank succeeded\n", rank);

    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
EOF

echo "=== Compiling simple test ==="
mpicxx -std=c++11 -O0 -g -Wall -I/usr/local/cuda-13.0/include -I/usr/include -o simple_test simple_test.cpp -L/usr/local/cuda-13.0/lib64 -L/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib -L/home/peng/projects/rdma_all/blue-rdma-driver/dtld-ibverbs/target/debug -L/usr/lib/x86_64-linux-gnu -lcudart -lnccl -lmpi -libverbs -lpthread -ldl

# Create GDB script
cat > gdb_script.txt << 'EOF'
set pagination off
set confirm off
run
bt
info proc mappings
quit
EOF

echo "=== Running simple test with GDB ==="
gdb -batch -x gdb_script.txt ./simple_test 2>&1 | tee gdb_simple.log

echo ""
echo "=== Running original test with single process ==="
# Try with single process to isolate the issue
mpirun -np 1 ./simple_test 2>&1 | tee single_process.log