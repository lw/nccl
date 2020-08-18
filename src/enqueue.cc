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

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;

  CUDACHECK(cudaGetDevice(&savedDev));

  struct ncclComm* comm = info->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  int rank = comm->rank;
  int nRanks = comm->nRanks;
  int peer = info->root;

  struct ncclInfo info2 = {
    /*coll=*/ncclCollSendRecv,
    /*opName=*/"SendRecv",
    /*sendbuff=*/info->sendbuff,
    /*recvbuff=*/info->recvbuff,
    /*length=*/info->length,
    /*root=*/-1,
    /*comm=*/comm,
    /*stream=*/info->stream
  };

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

  struct ncclColl coll;
  coll.args.sendbuff = info2.sendbuff;
  coll.args.recvbuff = info2.recvbuff;
  coll.args.comm = comm->devComm;

  coll.args.sendCount = info2.sendbytes;
  coll.args.recvCount = info2.recvbytes;
  coll.args.delta = info2.delta;
  coll.args.nThreads = info2.nThreads = comm->maxThreads + 2*WARP_SIZE;

  comm->myParams->blockDim.x = std::max<unsigned>(comm->myParams->blockDim.x, info2.nThreads);

  struct ncclChannel* channel = &comm->channel;

  comm->myParams->gridDim.x = 1;
  NCCLCHECK(ncclProxySaveP2p(&info2, channel));

  memcpy(&comm->args, &coll, sizeof(struct ncclColl));

  comm->myParams->func = (void*)ncclSendRecvKernel_copy_i8;

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
