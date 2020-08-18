/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "devcomm.h"
#include "primitives.h"

#define COLL_UNROLL 4

__device__ void ncclSendRecvKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-2*WARP_SIZE;

  // Compute pointers
  const int8_t* sendbuff = (const int8_t*)args->sendbuff;
  int8_t* recvbuff = (int8_t*)args->recvbuff;

  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;

  const int stepSize = comm->buffSize / NCCL_STEPS / SENDRECV_SLICEFACTOR;

  int nthreadsSplit = nthreads/2;

  if (tid < nthreadsSplit + WARP_SIZE ) {
    const ssize_t sendSize = args->sendCount;
    if (sendSize < 0) return;

    int peer = args->peer;
    SendPrimitive<COLL_UNROLL, int8_t>
      prims(tid, nthreadsSplit, peer, stepSize*4, channel, comm);

    if (sendSize == 0) {
      prims.send(sendbuff, 0);
    } else {
      for (ssize_t offset = 0; offset < sendSize; offset += stepSize) {
        int realChunkSize = min(stepSize, sendSize-offset);
        ALIGN_SIZE(realChunkSize, nthreads * sizeof(uint64_t));
        int nelem = min(realChunkSize, sendSize-offset);
        prims.send(sendbuff+offset, nelem);
      }
    }
  } else {
    const ssize_t recvSize = args->recvCount;
    if (recvSize < 0) return;

    int peer = args->peer;
    RecvPrimitive<COLL_UNROLL, int8_t>
      prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, peer, stepSize*4, channel, comm);

    if (recvSize == 0) {
      prims.recv(recvbuff, 0);
    } else {
      for (ssize_t offset = 0; offset < recvSize; offset += stepSize) {
        int realChunkSize = min(stepSize, recvSize-offset);
        ALIGN_SIZE(realChunkSize, nthreads * sizeof(uint64_t));
        int nelem = min(realChunkSize, recvSize-offset);
        prims.recv(recvbuff+offset, nelem);
      }
    }
  }
}
