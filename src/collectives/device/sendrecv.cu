/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sendrecv.h"
#include "common.h"
#include "collectives.h"

__device__ void ncclSendRecv_copy_i8(struct CollectiveArgs* args) {
  ncclSendRecvKernel<COLL_UNROLL, FuncSum<int8_t>, int8_t>(args);
}

__global__ void ncclSendRecvKernel_copy_i8(struct ncclColl firstColl) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE];
  ncclShmem = shmem;
  __shared__ struct ncclColl localColl;

  struct ncclDevComm* comm = firstColl.args.comm;
  struct ncclChannel* channel = comm->channels+bid;
  struct ncclColl* c;
  if (bid == 0) {
    /* To optimize for latency, (only) the first operation is passed as argument.*/
    c = &firstColl;
  } else {
    c = &localColl;
    load_coll(c, channel->collectives+channel->collFifoHead, tid, comm);
  }
  while (1) {
    if (tid < c->args.nThreads) {
      if (c->funcIndex == 0) {
        ncclSendRecvKernel<COLL_UNROLL, FuncSum<int8_t>, int8_t>(&c->args);
      } else {
        ncclFuncs[c->funcIndex](&c->args);
      }
    }
    int nextIndex = c->nextIndex;
    if (tid == 0) channel->collFifoHead = nextIndex;

    if (c->active == 2) {
      return;
    }

    /* Load next collective operation*/
    c = &localColl; /* for bid 0 */
    load_coll(c, channel->collectives+nextIndex, tid, comm);
  }
}
