/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"

#define MAX_ASYNC_OPS 128
thread_local pthread_t ncclGroupThreads[MAX_ASYNC_OPS];

#define NCCLCHECKTHREAD(a) do { \
  if ((args->ret = (a)) != ncclSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

static ncclResult_t scheduleSendRecv(struct ncclComm* comm, int delta, ssize_t recvbytes, void* recvbuff, ssize_t sendbytes, const void* sendbuff) {
  struct ncclInfo info = { ncclCollSendRecv, "SendRecv",
    sendbuff, recvbuff, (size_t)std::max<ssize_t>(sendbytes,recvbytes), ncclSum, -1, comm, comm->userStream, /* Args */
    1, 1 };
  info.delta = delta;
  info.sendbytes = sendbytes;
  info.recvbytes = recvbytes;
  if (delta == 0 && sendbytes != recvbytes) return ncclInvalidUsage;
  NCCLCHECK(ncclSaveKernel(&info));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupEnd, struct ncclComm*);
ncclResult_t ncclGroupEnd(struct ncclComm* comm) {
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  ncclResult_t ret = ncclSuccess;

  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct ncclP2Plist* p2plist = &comm->p2plist;
  if (p2plist->count) {
    for (int delta=0; delta<nRanks; delta++) {
      uint32_t from = (rank+nRanks-delta)%nRanks;
      uint32_t to = (rank+delta)%nRanks;

      ssize_t recvbytes = p2plist->peerlist[from].recvbytes;
      ssize_t sendbytes = p2plist->peerlist[to].sendbytes;
      p2plist->peerlist[from].recvbytes = -1;
      p2plist->peerlist[to].sendbytes = -1;
      if (sendbytes >= 0 || recvbytes >= 0) {
        NCCLCHECKGOTO(scheduleSendRecv(comm, delta,
              recvbytes, (char*)(p2plist->peerlist[from].recvbuff),
              sendbytes, (const char*)(p2plist->peerlist[to].sendbuff)),
          ret, end);
      }
    }
    p2plist->count = 0;
  }

  /* Collectives are done in three steps :
   * 1. Barrier Check In. Only the last call may call cudaLaunchKernel[cooperative]
   * 2. Barrier Wait. No CUDA call is permitted
   * 3. Enqueue Events. CUDA event wait/enqueue.
   * This is needed because step 2 cannot call any CUDA primitive, otherwise if
   * cudaFree happens between 1 and 3, it could block that CUDA call and
   * prevent some ranks from launching their network threads, which would
   * prevent the NCCL call from completing, blocking the cudaFree call.
   */

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ncclBarrierEnqueue(comm), ret, end);

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  NCCLCHECKGOTO(ncclBarrierEnqueueWait(comm), ret, end);

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ncclEnqueueEvents(comm), ret, end);

end:
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling cudaSetDevice, because this call can fail too
  return ret;
}
