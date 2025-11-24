#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <rccl/rccl.h>

int main(int argc, char *argv[]) {
  size_t size = 4;
  int root = 0;
  int ret;

  int rank, world_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  printf("rank: %d, world: %d\n", rank, world_size);

  // Set GPU device based on rank
  int num_devices;
  hipGetDeviceCount(&num_devices);
  int device = rank % num_devices;
  ret = hipSetDevice(device);
  printf("Rank %d using GPU %d, hipSetDevice() => %d\n", rank, device, ret);

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  hipStream_t s;

  if (rank == root) {
    ret = ncclGetUniqueId(&id);
    printf("ncclGetUniqueId() => %d\n", ret);
  }

  MPI_Bcast(&id, sizeof(id), MPI_BYTE, root, MPI_COMM_WORLD);

  ret = ncclCommInitRank(&comm, world_size, id, rank);
  printf("ncclCommInitRank() => %d\n", ret);

  ret = hipMalloc(&sendbuff, size * sizeof(float));
  printf("hipMalloc() => %d\n", ret);
  ret = hipMalloc(&recvbuff, size * sizeof(float));
  printf("hipMalloc() => %d\n", ret);
  ret = hipStreamCreate(&s);
  printf("hipStreamCreate() => %d\n", ret);

  float hostBuff[size];
  for (int i = 0; i < size; i++)
    hostBuff[i] = rank + 1;
  ret = hipMemcpy(sendbuff, hostBuff, size * sizeof(float),
                  hipMemcpyHostToDevice);
  printf("hipMemcpy() => %d\n", ret);

  ret = ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, s);
  printf("ncclReduce() => %d\n", ret);

  float result[size];
  ret =
      hipMemcpy(result, recvbuff, size * sizeof(float), hipMemcpyDeviceToHost);
  printf("hipMemcpy() => %d\n", ret);

  printf("result[0]: %f", result[0]);

  ret = hipStreamSynchronize(s);
  printf("hipStreamSynchronize() => %d\n", ret);

  ret = hipFree(sendbuff);
  printf("hipFree() => %d\n", ret);
  ret = hipFree(recvbuff);
  printf("hipFree() => %d\n", ret);
  ncclCommDestroy(comm);

  MPI_Finalize();

  /*ret = ncclGroupStart();*/
  /*printf("ncclGroupStart() => %d\n", ret);*/
  /*ncclGroupEnd();*/
  /**/
  /**/
  /**/
  /**/
  /**/
  /*ncclCommDestroy(comm);*/
  /**/
  /*printf("Sum = %f\n", result[0]);*/
  /**/
  return 0;
}
