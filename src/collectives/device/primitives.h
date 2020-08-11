/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <cassert>
#include <type_traits>
#include "copy.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

template <int UNROLL, typename T>
class SendPrimitive {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t sendStep;
  T* sendBuff;
  struct ncclDevComm* comm;

  inline __device__ int sendOffset() {
    return (sendStep%NCCL_STEPS)*stepSize;
  }

  inline __device__ T* sendPtr() {
    return ((T*)sendBuff)+sendOffset();
  }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 2, %0;" :: "r"(nthreads+WARP_SIZE));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync 4, %0;" :: "r"(nthreads));
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

  inline __device__ void incSend() {
    sendStep += 1;
  }

  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += 1;
  }

  inline __device__ void
  GenericOpSend(const T* srcPtr, int nelem) {
    int sliceSize = stepSize;
    int dataSize = max(DIVUP(nelem, 16)*16, sliceSize/32);

    const T* src = srcPtr;
    T* dst = sendPtr();

    bool syncThread = tid >= nthreads;

    int realSize = max(0, min(dataSize, nelem));
    if (!syncThread) {
      waitSend(realSize*sizeof(T));
      if (realSize > 0) {
        subBarrier();
        ReduceOrCopyMulti<UNROLL, T>(tid, nthreads, src, dst, realSize);
      }
    }
    barrier();

    incSend();

    if (syncThread) {
      if (realSize > 0 && wid == 0) __threadfence_system();
      __syncwarp();
      postSend();
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn) {
    sendBuff = (T*)conn->buff;
    sendStep = conn->step;
    if (wid == 0) sendConn = conn;
    if (wid == 0) sendConnTail = sendConnHead = sendStep; // Make sure we set this after rounding up
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid == 0) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
    }
    if (tid >= nthreads && wid == 0) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid == 0) {
      sendConn->step = sendConnHead;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  SendPrimitive(const int tid, const int nthreads, int sendPeer, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize) {
    // Make sure step is updated before we read it.
    barrier();

    loadSendConn(&channel->devPeers[sendPeer].send.conn);
    loadSendSync();
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOpSend(src, nelem);
  }

  __device__ __forceinline__ ~SendPrimitive() {
    // Save steps for the next operation
    saveSendSync();
  }
};

template <int UNROLL, typename T>
class RecvPrimitive {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;

  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  uint64_t recvStep;
  const T* recvBuff;
  struct ncclDevComm* comm;

  inline __device__ int recvOffset() {
    return (recvStep%NCCL_STEPS)*stepSize;
  }

  inline __device__ const T* recvPtr() {
    return ((const T*)recvBuff)+recvOffset();
  }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
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

  inline __device__ void incRecv() {
    recvStep += 1;
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void
  GenericOpRecv(T* dstPtr, int nelem) {
    int sliceSize = stepSize;
    int dataSize = max(DIVUP(nelem, 16)*16, sliceSize/32);

    const T* src = recvPtr();
    T* dst = dstPtr;

    bool syncThread = tid >= nthreads;

    int realSize = max(0, min(dataSize, nelem));
    if (!syncThread) {
      waitRecv();
      if (realSize > 0) {
        subBarrier();
        ReduceOrCopyMulti<UNROLL, T>(tid, nthreads, src, dst, realSize);
      }
    }
    barrier();

    incRecv();

    if (syncThread) {
      postRecv();
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn) {
    recvBuff = (const T*)conn->buff;
    recvStep = conn->step;
    if (wid == 0) recvConn = conn;
    if (wid == 0) recvConnTail = recvConnHead = recvStep; // Make sure we set this after rounding up
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2 * WARP_SIZE && wid == 0) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads && wid == 0) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads && wid == 0) {
      recvConn->step = recvConnHead;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  RecvPrimitive(const int tid, const int nthreads, int recvPeer, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize) {
    // Make sure step is updated before we read it.
    barrier();

    loadRecvConn(&channel->devPeers[recvPeer].recv.conn);
    loadRecvSync();
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOpRecv(dst, nelem);
  }

  __device__ __forceinline__ ~RecvPrimitive() {
    // Save steps for the next operation
    saveRecvSync();
  }
};

#endif
