/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

extern __device__ void ncclSendRecv_copy_i8(struct CollectiveArgs* args); \
extern __global__ void ncclSendRecvKernel_copy_i8(struct ncclColl c); \

// CHUNKSIZE must be a multiple of SLICESIZE
#define SENDRECV_SLICEFACTOR 4

#endif
