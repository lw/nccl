/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

extern __device__ volatile uint64_t* ncclShmem;

#define COLL_UNROLL 4

#endif
