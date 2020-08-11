/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "align.h"
#include <stdint.h>

typedef enum { ncclCollSendRecv } ncclFunc_t;
extern const char* ncclFuncStr[0];

#define NCCL_NUM_ALGORITHMS 2 // Tree/Ring
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

#define WARP_SIZE 32
#define MAXCHANNELS 32
#define NCCL_MAX_NTHREADS 512

#define NCCL_DIRECT_NIC 0x10

struct ncclConnInfo {
  // Regular comm mechanism
  char *buff; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication

  int *fifo;          // Size fifo for proxy

  uint64_t step;      // Keep where we are
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
  struct ncclComm *comm;
};


struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclDevComm;

/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclDevComm* comm;

  // local and remote input, output, and buffer
  const void * sendbuff;
  void * recvbuff;

  uint16_t nThreads;
  uint16_t unused;
  int32_t delta;
  size_t sendCount;
  size_t recvCount;
};
struct ncclColl {
  union {
    struct CollectiveArgs args;
    int data[0x10];
  };
};
static_assert(sizeof(struct ncclColl) == (0x10*sizeof(int)), "ncclColl must have a pow2 size");

struct ncclChannel {
  union {
    struct {
      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

typedef enum {
  ncclDevSuccess,
  ncclDevAssertedMismatch,
  ncclDevSuspectedMismatch
} ncclDevError_t;

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSize;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile ncclDevError_t *fatalDevError;

  // Channels, device side
  struct ncclChannel* channels;
};

#endif
