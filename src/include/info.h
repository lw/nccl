/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"

typedef enum {
  ncclPatternRing
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t length;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  // Computed later
  int nThreads;
  ssize_t sendbytes;
  ssize_t recvbytes;
  uint32_t delta;
};

#endif
