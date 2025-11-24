#include <cstdio>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>

int main(int argc, char *argv[])
{
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set GPU device based on rank
    int num_devices;
    hipGetDeviceCount(&num_devices);
    int device = rank % num_devices;
    hipSetDevice(device);
    printf("Rank %d/%d using GPU %d\n", rank, world_size, device);

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