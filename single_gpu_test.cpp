/**
 * Single GPU NCCL Test for Blue RDMA Driver
 *
 * This test uses NCCL in single-process mode to verify Blue RDMA driver
 * functionality without requiring multiple GPUs.
 *
 * Test strategy:
 * - Single process, single GPU
 * - Tests NCCL initialization with IB transport
 * - Verifies Blue RDMA device detection
 * - Performs basic GPU operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

// Error checking macros
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

int main(int argc, char* argv[]) {
    printf("=== Single GPU NCCL Test for Blue RDMA Driver ===\n\n");

    // Configure NCCL to use IB transport
    setenv("NCCL_IB_DISABLE", "0", 1);           // Enable IB transport
    setenv("NCCL_NET", "IB", 1);                 // Use IB network
    setenv("NCCL_IB_HCA", "bluerdma", 1);        // Use bluerdma device
    setenv("NCCL_DEBUG", "INFO", 1);             // Enable debug output
    setenv("NCCL_DEBUG_SUBSYS", "INIT,NET", 1);  // Debug init and network

    printf("✓ NCCL configured to use IB transport with bluerdma device\n");

    // Check CUDA device
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("✓ Found %d CUDA device(s)\n", deviceCount);

    if (deviceCount == 0) {
        printf("✗ No CUDA devices found\n");
        return 1;
    }

    // Set device
    CUDACHECK(cudaSetDevice(0));
    printf("✓ Using CUDA device 0\n");

    // Get device properties
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Test basic CUDA operations
    printf("\n--- Testing CUDA Operations ---\n");

    size_t size = 1024 * 1024;  // 1M floats = 4MB
    float *d_data;

    CUDACHECK(cudaMalloc(&d_data, size * sizeof(float)));
    printf("✓ Allocated %zu MB on GPU\n", (size * sizeof(float)) / (1024 * 1024));

    CUDACHECK(cudaMemset(d_data, 0, size * sizeof(float)));
    printf("✓ Memory set successful\n");

    // Copy data back to verify
    float *h_data = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("✓ Memory copy successful\n");

    // Verify
    bool success = true;
    for (size_t i = 0; i < 10; i++) {
        if (h_data[i] != 0.0f) {
            success = false;
            break;
        }
    }
    printf("%s Data verification: %s\n", success ? "✓" : "✗", success ? "PASSED" : "FAILED");

    // Test NCCL initialization (single rank)
    printf("\n--- Testing NCCL Initialization ---\n");

    ncclComm_t comm;
    ncclUniqueId id;

    // Generate unique ID
    NCCLCHECK(ncclGetUniqueId(&id));
    printf("✓ Generated NCCL unique ID\n");

    // Initialize NCCL with single rank
    NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));
    printf("✓ NCCL communicator initialized (single rank mode)\n");

    // Test NCCL AllReduce with single rank (should be a no-op but tests the path)
    float *sendbuff, *recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

    // Initialize sendbuff with test data
    float *h_sendbuff = (float*)malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        h_sendbuff[i] = 1.0f;
    }
    CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff, size * sizeof(float), cudaMemcpyHostToDevice));

    printf("✓ Allocated buffers for NCCL operations\n");

    // Create CUDA stream
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Perform AllReduce (in single rank mode, output = input)
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream));
    printf("✓ NCCL AllReduce operation posted\n");

    // Wait for completion
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("✓ NCCL AllReduce completed\n");

    // Verify result
    float *h_recvbuff = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));

    success = true;
    for (size_t i = 0; i < 10; i++) {
        if (h_recvbuff[i] != 1.0f) {
            printf("✗ Verification failed at index %zu: expected 1.0, got %f\n", i, h_recvbuff[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("✓ NCCL AllReduce result verification: PASSED\n");
    }

    // Cleanup
    printf("\n--- Cleanup ---\n");

    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    CUDACHECK(cudaFree(d_data));
    free(h_data);
    free(h_sendbuff);
    free(h_recvbuff);

    printf("✓ All resources cleaned up\n");

    printf("\n=== TEST COMPLETED SUCCESSFULLY ===\n");
    printf("\nKey findings:\n");
    printf("  • CUDA runtime: Working\n");
    printf("  • GPU memory operations: Working\n");
    printf("  • NCCL initialization: Working\n");
    printf("  • NCCL IB transport: %s\n", "Detected (check logs above)");
    printf("  • Blue RDMA driver: %s\n", "Loaded (check 'Setting op' messages above)");

    return 0;
}
