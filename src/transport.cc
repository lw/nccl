/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cassert>

#include "comm.h"
#include "info.h"
#include "bootstrap.h"

extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  netTransport,
};

template <int type>
static ncclResult_t selectTransport(struct ncclTopoSystem* topo, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(topo, myInfo, peerInfo, connect, connector));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, int peer) {
  struct ncclChannel* channel = &comm->channel;
  struct ncclConnect connect;
  struct ncclConnector* conn;

  {
    assert(peer != -1 && peer >= 0 && peer < comm->nRanks);
    conn = &channel->peers[peer].recv;
    assert(!conn->connected);
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<0>(comm->topo, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }

  {
    assert(peer != -1 && peer >= 0 && peer < comm->nRanks);
    conn = &channel->peers[peer].send;
    assert(!conn->connected);
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<1>(comm->topo, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }

  {
    assert(peer != -1 && peer >= 0 && peer < comm->nRanks);
    conn = &channel->peers[peer].send;
    assert(!conn->connected);
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, 1, comm->rank, conn));
    conn->connected = 1;
    CUDACHECK(cudaMemcpy(&channel->devPeers[peer].send, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
  }

  {
    assert(peer != -1 && peer >= 0 && peer < comm->nRanks);
    conn = &channel->peers[peer].recv;
    assert(!conn->connected);
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, 1, comm->rank, conn));
    conn->connected = 1;
    CUDACHECK(cudaMemcpy(&channel->devPeers[peer].recv, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
  }

  return ncclSuccess;
}
