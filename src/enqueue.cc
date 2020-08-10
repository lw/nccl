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
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclBarrierEnqueue(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams;

  if (comm->userStream != params->stream) {
    // Stream changed from last call, create dependency against last NCCL kernel launch
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  }
  params->stream = comm->userStream;

  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm) {
  struct cudaLaunchParams *params = comm->myParams;

  CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));

  params->gridDim.x = params->blockDim.x = 0;
  NCCLCHECK(ncclProxyStart(comm));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueEvents(ncclComm_t comm) {
  struct cudaLaunchParams *params = comm->myParams;
  // Enqueue event after NCCL kernel
  CUDACHECK(cudaEventRecord(comm->doneEvent, params->stream));
  comm->userStreamSet = false;
  return ncclSuccess;
}

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

static ncclResult_t checkSetStream(struct ncclInfo* info) {
 if (info->comm->userStreamSet == false) {
    info->comm->userStream = info->stream;
    info->comm->userStreamSet = true;
  } else if (info->stream != info->comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
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

  NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
  if (info->comm->checkPointers) {
    CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, end);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ArgsCheck(info), ret, end);
  NCCLCHECKGOTO(checkSetStream(info), ret, end);
end:
  if (savedDev != -1) CUDACHECK(cudaSetDevice(savedDev));

  struct ncclComm* comm = info->comm;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  int peer = info->root;

  struct ncclInfo info2 = { ncclCollSendRecv, "SendRecv",
    info->sendbuff, info->recvbuff, info->length, ncclSum, -1, comm, comm->userStream, /* Args */
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

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }

  NCCLCHECKGOTO(ncclBarrierEnqueue(comm), ret, end);

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  NCCLCHECKGOTO(ncclBarrierEnqueueWait(comm), ret, end);

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ncclEnqueueEvents(comm), ret, end);

  return ret;
}
