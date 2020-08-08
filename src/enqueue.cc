/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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
  // Only launch blocks where we have work to do.
  for (int c=0; c<comm->p2pnChannels; c++) {
    if (comm->channels[c].collCount) params->gridDim.x = c+1;
  }

  // Set active = 2 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<params->gridDim.x; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (channel->collCount == 0) {
      int opIndex = channel->collFifoTail;
      struct ncclColl* c = channel->collectives+opIndex;
      volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
      while (activePtr[0] != 0) sched_yield();

      c->args.p2p.delta = -1; // no-op
      c->funcIndex = 0;
      c->args.comm = comm->devComm;
      c->active = 1;
      opIndex = (opIndex+1)%NCCL_MAX_OPS;
      c->nextIndex = opIndex;
      channel->collFifoTail = opIndex;
      channel->collCount++;
    }
    channel->collectives[(channel->collStart+channel->collCount-1)%NCCL_MAX_OPS].active = 2;
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclColl* coll = comm->channels[0].collectives+comm->channels[0].collStart;
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
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->collStart = channel->collFifoTail;
    channel->collCount = 0;
  }
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

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclColl* coll, struct ncclProxyArgs* proxyArgs /* output */) {
  coll->args.sendbuff = info->sendbuff;
  coll->args.recvbuff = info->recvbuff;
  coll->args.comm = info->comm->devComm;
  coll->args.opCount = info->comm->opCount;

  coll->args.p2p.sendCount = info->sendbytes;
  coll->args.p2p.recvCount = info->recvbytes;
  coll->args.p2p.delta = info->delta;
  coll->funcIndex = 0;
  coll->args.p2p.nThreads = info->nThreads = info->comm->maxThreads + 2*WARP_SIZE;
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
  struct ncclProxyArgs proxyArgs;
  memset(&proxyArgs, 0, sizeof(struct ncclProxyArgs));
  NCCLCHECK(computeColl(info, &coll, &proxyArgs));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, info->nThreads);

  int channelId = info->channelId;
  struct ncclChannel* channel = info->comm->channels+channelId;

  if (channel->collCount == NCCL_MAX_OPS) {
    WARN("Too many aggregated operations on channel %d (%d max)", channel->id, NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }

  // Proxy
  proxyArgs.channel = channel;

  info->comm->myParams->gridDim.x = std::max<unsigned>(info->comm->myParams->gridDim.x, channelId+1);
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

// Save p2p operations in comm->p2plist. Operations will be posted to channels
// during ncclGroupEnd()
ncclResult_t ncclSaveP2p(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  struct ncclP2Plist* p2plist = &comm->p2plist;
  int peer = info->root;
  p2plist->count++;
  ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
  if (info->recvbuff == NULL) {
    if (peer != comm->rank) {
      int delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send.connected == 0) {
          p2plist->connect.send[channelId*comm->nRanks+p2plist->connect.nsend[channelId]++] = peer;
        }
      }
    }
    p2plist->peerlist[info->root].sendbytes = nBytes;
    p2plist->peerlist[info->root].sendbuff = info->sendbuff;
  } else {
    if (peer != comm->rank) {
      int delta = (comm->nRanks + (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv.connected == 0) {
          p2plist->connect.recv[channelId*comm->nRanks+p2plist->connect.nrecv[channelId]++] = peer;
        }
      }
    }
    p2plist->peerlist[info->root].recvbytes = nBytes;
    p2plist->peerlist[info->root].recvbuff = info->recvbuff;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    // Check arguments
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    if (info->comm->checkPointers) {
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, end);
    }
    NCCLCHECKGOTO(ArgsCheck(info), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECKGOTO(ncclAsyncColl(info->comm), ret, end);
    NCCLCHECKGOTO(checkSetStream(info), ret, end);

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    NCCLCHECKGOTO(ncclSaveP2p(info), ret, end);
end:
    if (savedDev != -1) CUDACHECK(cudaSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    NCCLCHECK(ArgsCheck(info));
    NCCLCHECK(checkSetStream(info));

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    NCCLCHECK(ncclSaveKernel(info));
    NCCLCHECK(ncclBarrierEnqueue(info->comm));
    NCCLCHECK(ncclBarrierEnqueueWait(info->comm));
    NCCLCHECK(ncclEnqueueEvents(info->comm));
    return ncclSuccess;
  }
}
