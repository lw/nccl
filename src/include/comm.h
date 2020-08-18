/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "transport.h"

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
    };
    char pad3[MEM_ALIGN];
  };
  char buff[1]; // Actually larger than that
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      int sizesFifo[NCCL_STEPS];
    };
    char pad4[MEM_ALIGN];
  };
  char buff[1]; // Actually larger than that
};

struct ncclComm {
  struct ncclChannel channel;

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  void* bootstrap;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format

  int node;
  int nNodes;
  int localRanks;

  // Buffer size
  int buffSize;

  // Algorithm/Protocols thresholds
  int maxThreads;

  // Whether there has been a fatal error in this communicator.
  ncclResult_t fatalError;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Device side of the communicator
  struct ncclDevComm *devComm;
  // Host copy of the devComm (to free CUDA allocs)
  struct ncclDevComm hostDevComm;

  // Global proxy thread
  pthread_t proxyThread;
  struct ncclProxyState proxyState;
};

#endif
