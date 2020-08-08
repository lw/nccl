/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t ncclTopoTuneModel(
  struct ncclComm* comm) {
  int simpleDefaultThreads = NCCL_MAX_NTHREADS;
  comm->maxThreads =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_MAX_NTHREADS, simpleDefaultThreads);

  return ncclSuccess;
}
