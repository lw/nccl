/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define FUNC_INDEX_P2P 0
#define FUNC_INDEX(coll, redop, dtype, al, pr) (1+(((((coll)*ncclNumOps + (redop))*ncclNumTypes) + (dtype))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_COLL_NAME(coll, op, dtype) \
  coll##_##op##_##dtype

#define NCCL_KERN_NAME(coll, op, dtype) \
  coll##Kernel_##op##_##dtype

/* Declare all collective operations */
#define DECL_COLL5(coll, op, dtype) \
  extern __device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl c); \

#define DECL_ALL_COLLS \
  DECL_COLL5(ncclSendRecv,copy,i8) \

DECL_ALL_COLLS

// CHUNKSIZE must be a multiple of SLICESIZE
#define SENDRECV_SLICEFACTOR 4

#endif
