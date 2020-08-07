/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"
#include <limits>

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

#endif // REDUCE_KERNEL_H_
