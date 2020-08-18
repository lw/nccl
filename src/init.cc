/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "net.h"
#include "graph.h"
#include "argcheck.h"
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[0] = { };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

ncclNet_t* ncclNet = NULL;

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initNet() {
  // Always initialize bootstrap network
  NCCLCHECK(bootstrapNetInit());

  if (ncclNet != NULL) return ncclSuccess;
  if (initNet(&ncclNetIb) == ncclSuccess) {
    ncclNet = &ncclNetIb;
  } else {
    NCCLCHECK(initNet(&ncclNetSocket));
    ncclNet = &ncclNetSocket;
  }
  return ncclSuccess;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    NCCLCHECK(initNet());
    INFO(NCCL_INIT, "Using network %s", ncclNetName());
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
}

#undef NCCL_NO_OPTIMIZE

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  free(comm->peerInfo);
  ncclTopoFree(comm->topo);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  CUDACHECK(cudaFree(comm->hostDevComm.channels));
  CUDACHECK(cudaFree(comm->devComm));

  NCCLCHECK(freeChannel(&comm->channel, comm->nRanks));

  CUDACHECK(cudaFreeHost((void *)comm->abortFlag));

  // Poison comm to try and catch a double free
  commPoison(comm);

  free(comm);
  return ncclSuccess;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  comm->rank = comm->hostDevComm.rank =rank;
  comm->nRanks = comm->hostDevComm.nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));

  comm->fatalError = ncclSuccess;

  NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
  comm->hostDevComm.abortFlag = comm->abortFlag;
  *comm->abortFlag = 0;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Duplicate the channels on the device
  NCCLCHECK(ncclCudaCalloc(&comm->hostDevComm.channels, 1));
  NCCLCHECK(ncclCudaMemcpy(comm->hostDevComm.channels, &comm->channel, 1));

  // Duplicate the dev comm on the device
  NCCLCHECK(ncclCudaCalloc(&comm->devComm, 1));
  NCCLCHECK(ncclCudaMemcpy(comm->devComm, &comm->hostDevComm, 1));
  return ncclSuccess;
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->hostHash=getHostHash()+commHash;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(&info->gdrSupport));
  return ncclSuccess;
}

#define DEFAULT_BUFFSIZE (1LL << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t env = ncclParamBuffSize();
  comm->buffSize = comm->hostDevComm.buffSize = env != -2 ? env : DEFAULT_BUFFSIZE;
  return ncclSuccess;
}

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 3 AllGathers
  // 1. { peerInfo, comm }
  // 2. ConnectTransport[nranks], ConnectValue[nranks]
  // 3. { nThreads, nrings, compCap, prev[MAXCHANNELS], next[MAXCHANNELS] }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap));

  // AllGather1 - begin
  struct {
    struct ncclPeerInfo peerInfo;
  } *allGather1Data;

  NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  struct ncclPeerInfo* myInfo = &allGather1Data[rank].peerInfo;
  NCCLCHECK(fillInfo(comm, myInfo, commHash));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather1Data, sizeof(*allGather1Data)));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks));
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
  }

  // Done with AllGather1 data
  free(allGather1Data);

  // AllGather1 - end

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  comm->nNodes = 2;
  comm->node = rank;

  NCCLCHECK(ncclTopoTuneModel(comm));

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
  NCCLCHECK(ncclTopoSetAffinity(comm->topo, comm->rank));
  ncclResult_t ret;

  NCCLCHECK(computeBuffSizes(comm));

  NCCLCHECKGOTO(initChannel(comm), ret, affinity_restore);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  if (comm->nNodes) NCCLCHECK(ncclProxyCreate(comm));

  return ncclSuccess;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;

  CUDACHECK(cudaSetDevice(cudaDev));
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %x - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId);

  return ncclSuccess;
cleanup:
  if ((*newcomm) && (*newcomm)->bootstrap) bootstrapAbort((*newcomm)->bootstrap);
  *newcomm = NULL;
  return res;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;

  NCCLCHECKGOTO(ncclInit(), res, end);

  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, end);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, end);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto end;
  }

  NCCLCHECKGOTO(ncclCommInitRankSync(newcomm, nranks, commId, myrank, cudaDev), res, end);
end:
  return res;
}

static ncclResult_t commDestroy(ncclComm_t comm) {
  int savedDevice;
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }


  NCCLCHECK(ncclProxyDestroy(comm));
  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(cudaSetDevice(savedDevice));


  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;


  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  return commDestroy(comm);
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;

  return commDestroy(comm);
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = comm->fatalError;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}
