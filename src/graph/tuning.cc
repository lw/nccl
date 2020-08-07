/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { { 4.4, 4.4,  0 }, { 3.6, 10.0, 8.4 }, { 4.4, 4.4,  0 } };

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
// Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
static const float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple)*/ { .52, 1.2, 28 }, /* Ring (LL/LL128/Simple)*/ { .47, 1.9, 3.4 }, /* CollNet (LL/LL128/Simple)*/ {  .5, 1.2, 4.0 } },
  /* PCI */
  { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9, 28 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 1.0, 1.9, 5.5 } },
  /* NET */
  { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 50 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 9.6 }, /* CollNet (LL/LL128/Simple)*/ { 5.0, 5.0, 10.7 } }
};

// LL128 max BW (per channel) for the different collectives
static const double ll128MaxBwPerCh[0] = { };

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph) {
  int simpleDefaultThreads = (ringGraph->speedIntra*ringGraph->nChannels <= PCI_WIDTH) ? 256 : NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);

  if (comm->nRanks <= 1) return ncclSuccess;

  int compCap80 = minCompCap == 80 && maxCompCap == 80 ? 1 : 0;
  float ppn = (float)comm->nRanks / comm->nNodes; // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetGraph };
  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = comm->nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1, 2, 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1, 1 };

  const char *protoStr = getenv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = getenv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  if (comm->rank == 0) {
    char line[1024];
    sprintf(line, "Latency/AlgBw |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %7s/%6s |", ncclAlgoStr[a], ncclProtoStr[p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    sprintf(line, " Max NThreads |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %14d |", comm->maxThreads[a][p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= comm->nRanks;

  // Override defaults with user env
  char* str = getenv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld/%ld/%ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 128MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][22] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .5,  .6,  .6,  .7,  .7,  .8,  .9,  .9, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .5,  .5,  .6,  .6,  .7,  .8,  .9 }
};

ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo* info, int algorithm, int protocol, float* time) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(info->nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && logSize < 22) bw *= treeCorrectionFactor[protocol][logSize];
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}
