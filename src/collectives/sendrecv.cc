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
  assert(comm->channel.peers[peer].send.connected);
  struct ncclInfo info = {
    /*sendbuff=*/sendbuff,
    /*recvbuff=*/NULL,
    /*sendbytes=*/static_cast<ssize_t>(count),
    /*recvbytes=*/-1,
    /*peer=*/peer,
    /*comm=*/comm,
    /*stream=*/stream};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  assert(comm->channel.peers[peer].recv.connected);
  struct ncclInfo info = {
    /*sendbuff=*/NULL,
    /*recvbuff=*/recvbuff,
    /*sendbytes=*/-1,
    /*recvbytes=*/static_cast<ssize_t>(count),
    /*peer=*/peer,
    /*comm=*/comm,
    /*stream=*/stream};
  return ncclEnqueueCheck(&info);
}
