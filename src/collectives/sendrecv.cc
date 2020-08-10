/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollSendRecv, "Send",
    sendbuff, NULL, count, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollSendRecv, "Recv",
    NULL, recvbuff, count, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}
