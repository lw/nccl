/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sendrecv.h"

__global__ void ncclSendRecvKernel_copy_i8(struct ncclColl firstColl) {
  if (threadIdx.x < firstColl.args.nThreads) {
    ncclSendRecvKernel(&firstColl.args);
  }
}
