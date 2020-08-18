/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"

// Used to pass NCCL call information between functions
struct ncclInfo {
  const void* sendbuff;
  void* recvbuff;
  ssize_t sendbytes;
  ssize_t recvbytes;
  int peer;
  ncclComm_t comm;
  cudaStream_t stream;
};

#endif
