/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sendrecv.h"
#include "common.h"
#include "collectives.h"

__global__ void ncclSendRecvKernel_copy_i8(struct ncclColl firstColl) {
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE];
  ncclShmem = shmem;

  if (threadIdx.x < firstColl.args.nThreads) {
    ncclSendRecvKernel<COLL_UNROLL, FuncSum<int8_t>, int8_t>(&firstColl.args);
  }
}
