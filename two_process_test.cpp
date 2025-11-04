/**
 * Two-Process NCCL Test for Blue RDMA Driver
 *
 * This test uses separate processes (not MPI) to avoid GPU duplication detection.
 * Each process uses the same GPU but different RDMA devices.
 *
 * Usage:
 *   Terminal 1: ./two_process_test 0    # Server (rank 0)
 *   Terminal 2: ./two_process_test 1    # Client (rank 1)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>

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

#define PORT 12345

// Exchange NCCL ID via TCP socket
void exchangeNcclId(int rank, ncclUniqueId* id) {
    if (rank == 0) {
        // Server: generate ID and send to client
        NCCLCHECK(ncclGetUniqueId(id));

        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(PORT);

        bind(server_fd, (struct sockaddr*)&address, sizeof(address));
        listen(server_fd, 1);

        printf("[Rank 0] Waiting for client connection on port %d...\n", PORT);

        int client_fd = accept(server_fd, NULL, NULL);
        send(client_fd, id, sizeof(ncclUniqueId), 0);

        printf("[Rank 0] NCCL ID sent to client\n");

        close(client_fd);
        close(server_fd);
    } else {
        // Client: receive ID from server
        printf("[Rank 1] Connecting to server on localhost:%d...\n", PORT);

        int sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_port = htons(PORT);
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);

        // Retry connection for up to 10 seconds
        int connected = 0;
        for (int i = 0; i < 20; i++) {
            if (connect(sock, (struct sockaddr*)&address, sizeof(address)) == 0) {
                connected = 1;
                break;
            }
            usleep(500000);  // 0.5 second
        }

        if (!connected) {
            printf("[Rank 1] Failed to connect to server\n");
            exit(1);
        }

        recv(sock, id, sizeof(ncclUniqueId), 0);
        printf("[Rank 1] NCCL ID received from server\n");

        close(sock);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <rank>\n", argv[0]);
        printf("  rank: 0 (server) or 1 (client)\n");
        return 1;
    }

    int rank = atoi(argv[1]);
    if (rank != 0 && rank != 1) {
        printf("Invalid rank: %d (must be 0 or 1)\n", rank);
        return 1;
    }

    printf("=== Two-Process NCCL Test for Blue RDMA Driver ===\n");
    printf("Process: Rank %d\n\n", rank);

    // Configure NCCL
    setenv("NCCL_IB_DISABLE", "0", 1);
    setenv("NCCL_NET", "IB", 1);

    // Use different RDMA device for each rank to avoid conflicts
    if (rank == 0) {
        setenv("NCCL_IB_HCA", "bluerdma0", 1);
        printf("[Rank 0] Using Blue RDMA device 0\n");
    } else {
        setenv("NCCL_IB_HCA", "bluerdma1", 1);
        printf("[Rank 1] Using Blue RDMA device 1\n");
    }

    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,NET", 1);

    // Initialize CUDA (both use GPU 0)
    CUDACHECK(cudaSetDevice(0));
    printf("[Rank %d] CUDA initialized on device 0\n", rank);

    // Exchange NCCL unique ID
    ncclUniqueId id;
    exchangeNcclId(rank, &id);

    // Initialize NCCL
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, 2, id, rank));
    printf("[Rank %d] NCCL communicator initialized\n", rank);

    // Allocate buffers
    size_t size = 1024 * 1024;  // 1M floats = 4MB
    float *sendbuff, *recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

    float *h_sendbuff = (float*)malloc(size * sizeof(float));
    float *h_recvbuff = (float*)malloc(size * sizeof(float));

    // Initialize data: rank 0 sends 1.0, rank 1 sends 2.0
    float send_value = (rank == 0) ? 1.0f : 2.0f;
    for (size_t i = 0; i < size; i++) {
        h_sendbuff[i] = send_value;
    }
    CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff, size * sizeof(float), cudaMemcpyHostToDevice));

    printf("[Rank %d] Initialized send buffer with value %.1f\n", rank, send_value);

    // Create CUDA stream
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Perform AllReduce (sum: 1.0 + 2.0 = 3.0)
    printf("[Rank %d] Starting AllReduce operation...\n", rank);
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream));

    // Wait for completion
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("[Rank %d] AllReduce completed\n", rank);

    // Copy result back and verify
    CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool success = true;
    float expected = 3.0f;  // 1.0 + 2.0
    for (size_t i = 0; i < 10; i++) {
        if (h_recvbuff[i] != expected) {
            printf("[Rank %d] ✗ Verification failed at index %zu: expected %.1f, got %.1f\n",
                   rank, i, expected, h_recvbuff[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[Rank %d] ✓ Test PASSED: result[0] = %.1f (expected %.1f)\n",
               rank, h_recvbuff[0], expected);
    } else {
        printf("[Rank %d] ✗ Test FAILED\n", rank);
    }

    // Cleanup
    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(h_sendbuff);
    free(h_recvbuff);

    printf("[Rank %d] Test completed\n", rank);

    return success ? 0 : 1;
}
