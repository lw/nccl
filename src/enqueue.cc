/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cassert>

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"

// Must be consistent with the ncclFuncSet enum
static void* const ncclKerns[1] = {
  (void*)ncclSendRecvKernel_copy_i8
};

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t setupLaunch(struct ncclComm* comm, struct cudaLaunchParams* params) {
    struct ncclChannel* channel = &comm->channel;

  // Only launch blocks where we have work to do.
  if (channel->collCount) {
    params->gridDim.x = 1;
  }

  // Set active = 2 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<params->gridDim.x; c++) {
    if (channel->collCount == 0) {
      int opIndex = channel->collFifoTail;
      struct ncclColl* c = channel->collectives+opIndex;
      volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
      while (activePtr[0] != 0) sched_yield();

      c->args.delta = -1; // no-op
      c->funcIndex = 0;
      c->args.comm = comm->devComm;
      c->active = 1;
      opIndex = (opIndex+1)%NCCL_MAX_OPS;
      c->nextIndex = opIndex;
      channel->collFifoTail = opIndex;
      channel->collCount++;
    }
    channel->collectives[(channel->collStart + channel->collCount - 1) % NCCL_MAX_OPS].active = 2;
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclColl* coll = comm->channel.collectives+comm->channel.collStart;
  memcpy(&comm->args, coll, sizeof(struct ncclColl));
  // As we pass that coll directly, we can free it immediately.
  coll->active = 0;

  params->func = ncclKerns[coll->funcIndex];
  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueue(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  NCCLCHECK(setupLaunch(comm, params));

  if (comm->userStream != params->stream) {
    // Stream changed from last call, create dependency against last NCCL kernel launch
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  }
  params->stream = comm->userStream;

  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm) {
  struct cudaLaunchParams *params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));

  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the ncclProxyStart call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  // for (int r=0; r<params->gridDim.x; r++) {  // FIXME gridDim
    struct ncclChannel* channel = &comm->channel;
    channel->collStart = channel->collFifoTail;
    channel->collCount = 0;
  // }
  params->gridDim.x = params->blockDim.x = 0;
  comm->lastOpCount = comm->opCount;
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
  coll->args.opCount = info->comm->opCount;

  coll->args.sendCount = info->sendbytes;
  coll->args.recvCount = info->recvbytes;
  coll->args.delta = info->delta;
  coll->funcIndex = 0;
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

  if (channel->collCount == NCCL_MAX_OPS) {
    WARN("Too many aggregated operations on channel (%d max)", NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }

  info->comm->myParams->gridDim.x = std::max<unsigned>(info->comm->myParams->gridDim.x, 1);
  NCCLCHECK(ncclProxySaveP2p(info, channel));
  info->comm->myParams->gridDim.x++;
  int opIndex = channel->collFifoTail;
  struct ncclColl* c = channel->collectives+opIndex;
  volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
  while (activePtr[0] != 0) sched_yield();

  memcpy(c, &coll, sizeof(struct ncclColl));

  c->active = 1;
  opIndex = (opIndex+1)%NCCL_MAX_OPS;
  c->nextIndex = opIndex;
  channel->collFifoTail = opIndex;
  channel->collCount++;

  info->comm->opCount++;
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
