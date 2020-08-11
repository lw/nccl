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

  if (args->delta < 0 ) return; // No-op

  if (args->delta == 0) {
    if (tid < nthreads && sendbuff != recvbuff) {
      // local copy : ReduceOrCopyMulti takes an int as number of elements,
      // so we split it in blocks of 1G elements.
      int blockSize = 1<<30;
      for (size_t offset=0; offset<args->sendCount; offset += blockSize) {
        size_t remaining = args->sendCount - offset;
        if (remaining < blockSize) blockSize = remaining;
        ReduceOrCopyMulti<COLL_UNROLL, int8_t>(tid, nthreads, 1, sendbuff, 1, recvbuff, blockSize);
        sendbuff += blockSize; recvbuff += blockSize;
      }
    }
    return;
  }

  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;

  const int stepSize = comm->buffSize / NCCL_STEPS / SENDRECV_SLICEFACTOR;

  int nthreadsSplit = nthreads/2;
  // We set NRECV or NSEND to 2 to use different barriers in primitives for the send threads and
  // receive threads, but then we define all peers to -1 since sender threads don't receive and
  // receive threads don't send.
  int peerNone[2] = {-1,-1};

  if (tid < nthreadsSplit + WARP_SIZE ) {
    const ssize_t sendSize = args->sendCount;
    if (sendSize < 0) return;

    int peer = (comm->rank+(int)args->delta)%comm->nRanks;
    ncclPrimitives<COLL_UNROLL, int8_t, /*NRECV=*/2, /*NSEND=*/1>
      prims(tid, nthreadsSplit, peerNone, &peer, recvbuff, stepSize*4, channel, comm);

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

    int peer = (comm->rank-(int)args->delta+comm->nRanks)%comm->nRanks;
    ncclPrimitives<COLL_UNROLL, int8_t, /*NRECV=*/1, /*NSEND=*/2>
      prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &peer, peerNone, recvbuff, stepSize*4, channel, comm);

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
