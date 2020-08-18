/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cassert>

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclColl* coll) {
  coll->args.sendbuff = info->sendbuff;
  coll->args.recvbuff = info->recvbuff;
  coll->args.comm = info->comm->devComm;

  coll->args.sendCount = info->sendbytes;
  coll->args.recvCount = info->recvbytes;
  coll->args.delta = info->delta;
  coll->args.nThreads = info->nThreads = info->comm->maxThreads + 2*WARP_SIZE;
  return ncclSuccess;
}

ncclResult_t ncclSaveKernel(struct ncclInfo* info) {
  struct ncclColl coll;
  NCCLCHECK(computeColl(info, &coll));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, info->nThreads);

  struct ncclChannel* channel = &info->comm->channel;

  info->comm->myParams->gridDim.x = 1;
  NCCLCHECK(ncclProxySaveP2p(info, channel));

  memcpy(&info->comm->args, &coll, sizeof(struct ncclColl));

  info->comm->myParams->func = (void*)ncclSendRecvKernel_copy_i8;

  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;

  CUDACHECK(cudaGetDevice(&savedDev));

  struct ncclComm* comm = info->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  int rank = comm->rank;
  int nRanks = comm->nRanks;
  int peer = info->root;

  struct ncclInfo info2 = { ncclCollSendRecv, "SendRecv",
    info->sendbuff, info->recvbuff, info->length, -1, comm, info->stream, /* Args */
    1, 1 };

  if (info->recvbuff == NULL) {
    if (peer != rank) {
      assert(comm->channel.peers[peer].send.connected);
    }
    info2.delta = (peer - rank + nRanks) % nRanks;
    info2.sendbytes = info->length;
  } else {
    if (peer != comm->rank) {
      assert(comm->channel.peers[peer].recv.connected);
    }
    info2.delta = (rank - peer + nRanks) % nRanks;
    info2.recvbytes = info->length;
  }

  NCCLCHECK(ncclSaveKernel(&info2));

  struct cudaLaunchParams* params = comm->myParams;
  params->stream = info->stream;

  CUDACHECKGOTO(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream), ret, end);

  params->gridDim.x = params->blockDim.x = 0;
  NCCLCHECKGOTO(ncclProxyStart(comm), ret, end);

end:
  if (savedDev != -1) {
    CUDACHECK(cudaSetDevice(savedDev));
  }

  return ret;
}
