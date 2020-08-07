/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common.h"

__device__ volatile uint64_t* ncclShmem;

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_COLL_NAME(ncclSendRecv, copy, i8) }

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ncclFuncs[1] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_COLL_NAME(ncclSendRecv, copy, i8)
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
