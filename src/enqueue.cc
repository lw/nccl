/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cassert>

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;

  CUDACHECK(cudaGetDevice(&savedDev));

  struct ncclComm* comm = info->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  struct ncclChannel* channel = &comm->channel;
  NCCLCHECK(ncclProxySaveP2p(info, channel));

  struct ncclColl coll;
  // TODO memset to zero?
  coll.args.sendbuff = info->sendbuff;
  coll.args.recvbuff = info->recvbuff;
  coll.args.comm = comm->devComm;
  coll.args.sendCount = info->sendbytes;
  coll.args.recvCount = info->recvbytes;
  coll.args.peer = info->peer;
  coll.args.nThreads = comm->maxThreads + 2 * WARP_SIZE;

  void* collPtr = &coll;
  CUDACHECKGOTO(cudaLaunchKernel(
    /*func=*/(void*)ncclSendRecvKernel_copy_i8,
    /*gridDim=*/dim3(),
    /*blockDim=*/dim3(coll.args.nThreads),
    /*args=*/&collPtr,
    /*sharedMem=*/0,
    /*stream=*/info->stream),
  ret, end);

  NCCLCHECKGOTO(ncclProxyStart(comm), ret, end);

end:
  if (savedDev != -1) {
    CUDACHECK(cudaSetDevice(savedDev));
  }

  return ret;
}
