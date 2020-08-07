/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 13, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}

typedef void(*ncclKern_t)(struct CollectiveArgs* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid, struct ncclDevComm* comm) {
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort);
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid);
  __syncthreads();
  if (tid == 0) hostColl->active = 0;
}

extern __device__ volatile uint64_t* ncclShmem;

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(args); \
}

/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE]; \
  ncclShmem = shmem; \
  __shared__ struct ncclColl localColl; \
 \
  struct ncclDevComm* comm = firstColl.args.comm; \
  struct ncclChannel* channel = comm->channels+bid; \
  struct ncclColl* c; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, channel->collectives+channel->collFifoHead, tid, comm); \
  } \
  while (1) { \
    if (tid < c->args.common.nThreads) { \
      if (c->funcIndex == fIndex) { \
        coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(&c->args); \
      } else { \
        ncclFuncs[c->funcIndex](&c->args); \
      } \
    } \
    int nextIndex = c->nextIndex; \
    if (tid == 0) channel->collFifoHead = nextIndex; \
 \
    if (c->active == 2) { \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    c = &localColl; /* for bid 0 */ \
    load_coll(c, channel->collectives+nextIndex, tid, comm); \
  } \
}

#define COLL_UNROLL 4

#endif
