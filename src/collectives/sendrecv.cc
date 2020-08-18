/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cassert>

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO Device guard

  assert(count > 0);

  {
    struct ncclChannel* channel = &comm->channel;
    struct ncclProxyArgs args;
    memset(&args, 0, sizeof(args));
    args.channel = channel;
    args.nsteps = DIVUP(count, comm->buffSize/NCCL_STEPS/SENDRECV_SLICEFACTOR);

    struct ncclConnector* connector = &channel->peers[peer].send;
    assert(connector->connected);
    if (connector->transportComm == NULL) {
      WARN("[%d] Error no transport for send peer %d on channel\n", connector->comm->rank,
          peer);
      return ncclInternalError;
    }
    if (connector->transportComm->proxy != NULL) {
      NCCLCHECK(SaveProxy(connector, &args));
    }
  }

  struct ncclColl coll;
  // TODO memset to zero?
  coll.args.sendbuff = sendbuff;
  coll.args.recvbuff = NULL;
  coll.args.comm = comm->devComm;
  coll.args.sendCount = static_cast<ssize_t>(count);
  coll.args.recvCount = -1;
  coll.args.peer = peer;
  coll.args.nThreads = comm->maxThreads + 2 * WARP_SIZE;

  void* collPtr = &coll;
  CUDACHECK(cudaLaunchKernel(
    /*func=*/(void*)ncclSendRecvKernel_copy_i8,
    /*gridDim=*/dim3(),
    /*blockDim=*/dim3(coll.args.nThreads),
    /*args=*/&collPtr,
    /*sharedMem=*/0,
    /*stream=*/stream));

  NCCLCHECK(ncclProxyStart(comm));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO Device guard

  assert(count > 0);

  {
    struct ncclChannel* channel = &comm->channel;
    struct ncclProxyArgs args;
    memset(&args, 0, sizeof(args));
    args.channel = channel;
    args.nsteps = DIVUP(count, comm->buffSize/NCCL_STEPS/SENDRECV_SLICEFACTOR);

    struct ncclConnector* connector = &channel->peers[peer].recv;
    assert(connector->connected);
    if (connector->transportComm == NULL) {
      WARN("[%d] Error no transport for recv peer %d on channel\n", connector->comm->rank,
          peer);
      return ncclInternalError;
    }
    if (connector->transportComm->proxy != NULL) {
      NCCLCHECK(SaveProxy(connector, &args));
    }
  }

  struct ncclColl coll;
  // TODO memset to zero?
  coll.args.sendbuff = NULL;
  coll.args.recvbuff = recvbuff;
  coll.args.comm = comm->devComm;
  coll.args.sendCount = -1;
  coll.args.recvCount = static_cast<ssize_t>(count);
  coll.args.peer = peer;
  coll.args.nThreads = comm->maxThreads + 2 * WARP_SIZE;

  void* collPtr = &coll;
  CUDACHECK(cudaLaunchKernel(
    /*func=*/(void*)ncclSendRecvKernel_copy_i8,
    /*gridDim=*/dim3(),
    /*blockDim=*/dim3(coll.args.nThreads),
    /*args=*/&collPtr,
    /*sharedMem=*/0,
    /*stream=*/stream));

  NCCLCHECK(ncclProxyStart(comm));

  return ncclSuccess;
}
