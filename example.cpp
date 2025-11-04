#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char *argv[])
{
    size_t size = 1024;
    int root = 0;

    // Initialize MPI
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("[Rank %d/%d] Process started (PID: %d)\n", rank, world_size, getpid());

    // Configure NCCL to use IB transport
    // This ensures NCCL will use your Blue RDMA driver
    setenv("NCCL_IB_DISABLE", "0", 1);           // Enable IB transport
    setenv("NCCL_NET", "IB", 1);                 // Use IB network
    setenv("NCCL_IB_HCA", "bluerdma", 1);       // Use bluerdma device
    setenv("NCCL_DEBUG", "INFO", 1);             // Enable debug output
    setenv("NCCL_DEBUG_SUBSYS", "INIT,NET", 1);  // Debug init and network

    printf("[Rank %d] NCCL configured to use IB transport with bluerdma device\n", rank);

    // Get NCCL unique ID
    ncclUniqueId id;
    if (rank == root) {
        NCCLCHECK(ncclGetUniqueId(&id));
        printf("[Rank %d] Generated NCCL unique ID\n", rank);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, root, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    ncclComm_t comm;
    // For single GPU testing: all processes share GPU 0
    // For multi-GPU: use cudaSetDevice(rank % num_gpus)
    CUDACHECK(cudaSetDevice(0));  // Use GPU 0 for all processes
    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));
    printf("[Rank %d] NCCL communicator initialized\n", rank);

    // Allocate GPU memory
    float *sendbuff, *recvbuff;
    cudaStream_t stream;

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&stream));
    printf("[Rank %d] GPU memory allocated (%zu bytes)\n", rank, size * sizeof(float));

    // Initialize send buffer
    float *hostBuff = new float[size];
    for (int i = 0; i < size; i++) {
        hostBuff[i] = rank + 1.0f;
    }
    CUDACHECK(cudaMemcpy(sendbuff, hostBuff, size * sizeof(float), cudaMemcpyHostToDevice));
    printf("[Rank %d] Send buffer initialized with value %.1f\n", rank, hostBuff[0]);

    // Perform AllReduce operation
    printf("[Rank %d] Starting ncclAllReduce...\n", rank);
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("[Rank %d] ncclAllReduce completed\n", rank);

    // Copy result back to host
    float *result = new float[size];
    CUDACHECK(cudaMemcpy(result, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    float expected_sum = (world_size * (world_size + 1)) / 2.0f;
    bool success = true;
    for (int i = 0; i < size; i++) {
        if (result[i] != expected_sum) {
            printf("[Rank %d] ERROR: result[%d] = %.1f, expected %.1f\n",
                   rank, i, result[i], expected_sum);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[Rank %d] ✓ Test PASSED: result[0] = %.1f (expected %.1f)\n",
               rank, result[0], expected_sum);
    }

    // Cleanup
    delete[] hostBuff;
    delete[] result;
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);

    printf("[Rank %d] Cleanup completed\n", rank);

    MPI_Finalize();

    printf("[Rank %d] Test finished successfully\n", rank);
    return 0;
}
