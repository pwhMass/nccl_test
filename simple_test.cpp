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
