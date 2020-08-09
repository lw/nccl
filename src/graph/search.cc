/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "xml.h"
#include <math.h>

// Initialize system->maxWidth. This is the per-channel (i.e. per-SM)
// max speed.
static float getMaxWidth(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxWidth = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    float width = path->width;
    if (path->count == 0) continue;
    maxWidth = std::max(maxWidth, width);
  }
  return maxWidth;
}

ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxWidth = 0.0;
  int inter = system->nodes[NET].count;
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxWidth = LOC_WIDTH;
    return ncclSuccess;
  }
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    system->maxWidth = std::max(system->maxWidth, getMaxWidth(system, gpu, inter ? NET : GPU));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetDev(struct ncclTopoSystem* system, int rank, int channelId, int* dev) {
  int64_t id;
  NCCLCHECK(ncclTopoGetLocalNet(system, rank, &id, channelId));
  *dev = id;

  return ncclSuccess;
}
