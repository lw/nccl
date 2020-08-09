/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"

#define MAX_ASYNC_OPS 128
thread_local pthread_t ncclGroupThreads[MAX_ASYNC_OPS];
thread_local int ncclGroupIndex = 0;
thread_local int ncclGroupMode = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;

bool ncclAsyncMode() {
  return ncclGroupMode > 0;
}

ncclResult_t ncclAsyncErrCheck(ncclResult_t ret) {
  if (ncclGroupError == ncclSuccess || ret != ncclSuccess) ncclGroupError = ret;
  return ret;
}

struct ncclCollArgs {
  ncclComm_t comm;
  int connect;
};

enum ncclAsyncFuncType {
  ASYNC_FUNC_INVALID = 0,
  ASYNC_FUNC_COLL = 1,
};
struct ncclAsyncArgs {
  ncclResult_t ret;
  enum ncclAsyncFuncType funcType;
  ncclCollArgs coll;
};

thread_local struct ncclAsyncArgs ncclGroupArgs[MAX_ASYNC_OPS];

#define NCCLCHECKTHREAD(a) do { \
  if ((args->ret = (a)) != ncclSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

ncclResult_t ncclAsyncColl(ncclComm_t comm) {
  struct ncclAsyncArgs* args = ncclGroupArgs;
  for (int i=0; i<ncclGroupIndex; i++) {
    if (args->coll.comm == comm) return ncclSuccess;
    args++;
  }
  if (ncclGroupIndex >= MAX_ASYNC_OPS) {
    WARN("Too many async operations in progress, max is %d", MAX_ASYNC_OPS);
    return ncclAsyncErrCheck(ncclInvalidUsage);
  }
  ncclGroupIndex++;
  args->funcType = ASYNC_FUNC_COLL;
  args->coll.comm = comm;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  if (ncclGroupMode == 0) {
    memset(ncclGroupArgs, 0, sizeof(struct ncclAsyncArgs)*MAX_ASYNC_OPS);
  }
  ncclGroupMode++;
  return ncclSuccess;
}

static ncclResult_t scheduleSendRecv(struct ncclComm* comm, int delta, ssize_t recvbytes, void* recvbuff, ssize_t sendbytes, const void* sendbuff) {
  struct ncclInfo info = { ncclCollSendRecv, "SendRecv",
    sendbuff, recvbuff, (size_t)std::max<ssize_t>(sendbytes,recvbytes), ncclInt8, ncclSum, -1, comm, comm->userStream, /* Args */
    1, 1 };
  info.delta = delta;
  info.sendbytes = sendbytes;
  info.recvbytes = recvbytes;
  if (delta == 0 && sendbytes != recvbytes) return ncclInvalidUsage;
  NCCLCHECK(ncclSaveKernel(&info));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  if (ncclGroupMode == 0) return ncclInvalidUsage;
  ncclGroupMode--;
  if (ncclGroupMode > 0) return ncclSuccess;
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  ncclResult_t ret = ncclGroupError;
  if (ret != ncclSuccess) goto group_cleanup;

  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      struct ncclComm* comm = args->coll.comm;
      int rank = comm->rank;
      int nRanks = comm->nRanks;
      struct ncclP2Plist* p2plist = &args->coll.comm->p2plist;
      if (p2plist->count) {
        for (int delta=0; delta<nRanks; delta++) {
          uint32_t from = (rank+nRanks-delta)%nRanks;
          uint32_t to = (rank+delta)%nRanks;

          // Compute how much to split operations
          // Natural step size matching buffer steps.
          ssize_t stepSize = 4*comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
          ssize_t recvChunkSize = p2plist->peerlist[from].recvbytes;
          ssize_t sendChunkSize = p2plist->peerlist[to].sendbytes;
          recvChunkSize = std::max((ssize_t)1, DIVUP(recvChunkSize, stepSize)) * stepSize;
          sendChunkSize = std::max((ssize_t)1, DIVUP(sendChunkSize, stepSize)) * stepSize;

          ssize_t sendOffset = 0;
          ssize_t recvOffset = 0;
          int remaining = 1;
          int chunk = 0;
          while (remaining) {
            remaining = 0;
            ssize_t recvbytes = p2plist->peerlist[from].recvbytes-recvOffset;
            ssize_t sendbytes = p2plist->peerlist[to].sendbytes-sendOffset;
            if (recvbytes > recvChunkSize) { remaining = 1; recvbytes = recvChunkSize; } else p2plist->peerlist[from].recvbytes = -1;
            if (sendbytes > sendChunkSize) { remaining = 1; sendbytes = sendChunkSize; } else p2plist->peerlist[to].sendbytes = -1;
            if (sendbytes >= 0 || recvbytes >= 0) {
              NCCLCHECKGOTO(scheduleSendRecv(comm, delta,
                    recvbytes, ((char*)(p2plist->peerlist[from].recvbuff)) + recvOffset,
                    sendbytes, ((const char*)(p2plist->peerlist[to].sendbuff)) + sendOffset), ret, end);
            }
            recvOffset += recvChunkSize;
            sendOffset += sendChunkSize;
            chunk++;
          }
        }
        p2plist->count = 0;
      }
    }
  }

  /* Collectives are done in three steps :
   * 1. Barrier Check In. Only the last call may call cudaLaunchKernel[cooperative]
   * 2. Barrier Wait. No CUDA call is permitted
   * 3. Enqueue Events. CUDA event wait/enqueue.
   * This is needed because step 2 cannot call any CUDA primitive, otherwise if
   * cudaFree happens between 1 and 3, it could block that CUDA call and
   * prevent some ranks from launching their network threads, which would
   * prevent the NCCL call from completing, blocking the cudaFree call.
   */
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      if (args->coll.comm->userStream == NULL)
        CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclBarrierEnqueue(args->coll.comm), ret, end);
    }
  }
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclBarrierEnqueueWait(args->coll.comm), ret, end);
    }
  }
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      if (args->coll.comm->userStream == NULL)
        CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclEnqueueEvents(args->coll.comm), ret, end);
    }
  }

  goto end;
group_cleanup:
  if (ret != ncclSuccess) {
    // At least one call in the group failed. Since we want to make that group
    // an atomic operation, we need to cancel all operations.
    for (int i=0; i<ncclGroupIndex; i++) {
      struct ncclAsyncArgs* args = ncclGroupArgs+i;
      struct ncclComm* comm = args->coll.comm;
      struct ncclChannel* channel = &comm->channel;
      for (int i=0; i<channel->collCount; i++) {
        channel->collectives[(channel->collStart + i)%NCCL_MAX_OPS].active = 0;
      }
      channel->collFifoTail = channel->collStart;
      channel->collCount = 0;
      /* Cancel all proxy ops : mark them as ncclProxyOpNone and they should be freed later on */
      struct ncclProxyState* state = &comm->proxyState;
      struct ncclProxyArgs *op, *start;
      pthread_mutex_lock(&state->mutex);
      op = start = state->ops;
      while (op) {
        if (op->opCount >= comm->lastOpCount) op->state = ncclProxyOpNone;
        struct ncclProxyArgs* peerOp = op->nextPeer;
        while (peerOp) {
          if (peerOp->opCount >= comm->lastOpCount) peerOp->state = ncclProxyOpNone;
          peerOp = peerOp->nextPeer;
        }
        op = op->next;
        if (op == start) break;
      }
      comm->opCount = comm->lastOpCount;
      pthread_cond_signal(&state->cond);
      pthread_mutex_unlock(&state->mutex);

      comm->myParams->gridDim.x = comm->myParams->blockDim.x = 0;
      comm->userStreamSet = false;
    }
  }
end:
  ncclGroupError = ncclSuccess;
  ncclGroupIndex = 0;
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling cudaSetDevice, because this call can fail too
  return ret;
}
