#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if(e != cudaSuccess) {                            \
    std::cerr << "Failed: Cuda error " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(e) << "'" << std::endl; \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    std::cerr << "Failed, NCCL error " << __FILE__ << ":" << __LINE__ << " '" << ncclGetErrorString(r) << "'" << std::endl; \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
  int world_size = 2;
  int size = 32 * 1024 * 1024;
  pid_t childPids[2];
  int fds[2];

  int rv = pipe(fds);
  assert(rv == 0);

  for (int rank = 0; rank < world_size; rank++) {
    childPids[rank] = fork();

    if (childPids[rank] != 0) {
      std::cout << "Child " << rank << " has PID " << childPids[rank] << std::endl;
      continue;
    }

    ncclUniqueId id;
    if (rank == 0) {
      close(fds[0]);
      NCCLCHECK(ncclGetUniqueId(&id));
      int nbytes = write(fds[1], &id, sizeof(id));
      assert(nbytes == sizeof(id));
    } else {
      close(fds[1]);
      int nbytes = read(fds[0], &id, sizeof(id));
      assert(nbytes == sizeof(id));
    }

    float* sendbuff;
    float* recvbuff;
    cudaStream_t s;

    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    ncclComm_t comm;

    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));

    ncclTransportP2pSetup(comm, rank ^ 1);

    if (rank == 0) {
      NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, rank ^ 1, comm, s));
    } else {
      NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, rank ^ 1, comm, s));
    }

    CUDACHECK(cudaStreamSynchronize(s));

    if (rank == 1) {
      std::unique_ptr<uint8_t[]> cpuPtr((uint8_t*)malloc(size * sizeof(float)));
      CUDACHECK(cudaMemcpy((void*)cpuPtr.get(), (const void*)recvbuff, size * sizeof(float), cudaMemcpyDefault));
      for (int i = 0; i < size * sizeof(float); i ++) {
        assert(cpuPtr[i] == 1);
      }
    }

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    ncclCommDestroy(comm);

    std::cout << "Child " << rank << " success" << std::endl;
    return 0;
  }

  close(fds[0]);
  close(fds[1]);

  for (int rank = 0; rank < world_size; rank++) {
    pid_t childPid = wait(nullptr);
    std::cout << "Child " << rank << " (PID " << childPid << ") returned" << std::endl;
  }

  return 0;
}
