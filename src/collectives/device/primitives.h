/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Implementation of primitive types
template <int UNROLL, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) {
    return (recvStep[i]%NCCL_STEPS)*stepSize;
  }

  inline __device__ int sendOffset(int i) {
    return (sendStep[i]%NCCL_STEPS)*stepSize;
  }

  inline __device__ const T* recvPtr(int i) {
    return ((const T*)recvBuff[i])+recvOffset(i);
  }

  inline __device__ T* sendPtr(int i) {
    return ((T*)sendBuff[i])+sendOffset(i);
  }

  inline __device__ void barrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads+WARP_SIZE));
    }
  }
  inline __device__ void subBarrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 4, %0;" :: "r"(nthreads));
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + 1) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      recvConnTail += 1;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += 1;
  }

  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += 1;
  }

  inline __device__ void
  GenericOpSend(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize;
    int dataSize = max(DIVUP(nelem, 16)*16, sliceSize/32);

    const T* srcs[1];
    srcs[0] = srcPtr;

    T* dsts[NSEND];
    dsts[0] = sendPtr(0);
    for (int i=1; i<NSEND && i<nsend; i++) dsts[i] = sendPtr(i);

    bool syncThread = tid >= nthreads;

    int realSize = max(0, min(dataSize, nelem-offset));
    if (!syncThread) {
      waitSend(realSize*sizeof(T));
      if (realSize > 0) {
        subBarrier();
        static_assert(NSEND == 1, "!");
        ReduceOrCopyMulti<UNROLL, FUNC, T>(tid, nthreads, 1, srcs[0], nsend, dsts[0], realSize);
      }
    }
    barrier();

    /* Send to far first, then close */
    for (int i=1; i<NSEND && i<nsend; i++) incSend(i);
    incSend(0);

    if (syncThread) {
      if (realSize > 0 && wid == 0) __threadfence_system();
      __syncwarp();
      postSend();
    }
    srcs[0] += realSize;
    dsts[0] += sliceSize;
    for (int i=1; i<NSEND; i++) dsts[i] += sliceSize;
    offset += realSize;
  }

  inline __device__ void
  GenericOpRecv(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize;
    int dataSize = max(DIVUP(nelem, 16)*16, sliceSize/32);

    const T* srcs[NRECV];
    srcs[0] = recvPtr(0);
    for (int i=1; i<NRECV && i<nrecv; i++) srcs[i] = recvPtr(i);

    T* dsts[1];
    dsts[0] = dstPtr;

    bool syncThread = tid >= nthreads;

    int realSize = max(0, min(dataSize, nelem-offset));
    if (!syncThread) {
      waitRecv();
      if (realSize > 0) {
        subBarrier();
        static_assert(NRECV == 1, "!");
        ReduceOrCopyMulti<UNROLL, FUNC, T>(tid, nthreads, nrecv, srcs[0], 1, dsts[0], realSize);
      }
    }
    barrier();

    /* Recv from close first, then far */
    incRecv(0);
    for (int i=1; i<NRECV && i<nrecv; i++) incRecv(i);

    if (syncThread) {
      postRecv();
    }
    srcs[0] += sliceSize;
    for (int i=1; i<NRECV; i++) srcs[i] += sliceSize;
    dsts[0] += realSize;
    offset += realSize;
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvBuff[i] = (const T*)conn->buff;
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (T*)conn->buff;
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
    }
    if (tid >= nthreads && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads && wid < nrecv) {
      recvConn->step = recvConnHead;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOpSend(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOpRecv(NULL, dst, nelem, 0);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#endif
