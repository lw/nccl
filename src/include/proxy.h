/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include <pthread.h>

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  struct ncclChannel* channel;
  struct ncclConnector* connector;
  int nsteps;
  int state;   // add component before this line -- it is left out during initialization

  // Internal state
  uint64_t head;
  uint64_t tail;
  uint64_t end;
  void* requests[NCCL_STEPS];
  int idle;

  // Element linking
  // pthread_mutex_t mutex;
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
};

#define PROXYARGS_ALLOCATE_SIZE 32
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

struct ncclProxyState {
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  bool stop;
  struct ncclProxyArgs* ops;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveP2p(struct ncclInfo* info, struct ncclChannel* channel);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);

#include <unistd.h>

#endif
