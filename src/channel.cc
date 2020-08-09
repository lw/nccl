/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"

ncclResult_t initChannel(struct ncclComm* comm) {
  struct ncclChannel* channel = &comm->channel;

  // Communication structures with peers.
  NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks));
  for (size_t i=0; i<comm->nRanks; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
  }

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostCalloc(&channel->collectives, NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->collectives));

  // Free transport proxy resources
  for (int r=0; r<nRanks; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->send.transportResources) NCCLCHECK(peer->send.transportComm->free(peer->send.transportResources));
  }
  for (int r=0; r<nRanks; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->recv.transportResources) NCCLCHECK(peer->recv.transportComm->free(peer->recv.transportResources));
  }

  // Free the peer structures.
  CUDACHECK(cudaFree(channel->devPeers));
  free(channel->peers);

  return ncclSuccess;
}
