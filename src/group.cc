/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"

NCCL_API(ncclResult_t, ncclGroupEnd, struct ncclComm*);
ncclResult_t ncclGroupEnd(struct ncclComm* comm) {
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  ncclResult_t ret = ncclSuccess;

  /* Collectives are done in three steps :
   * 1. Barrier Check In. Only the last call may call cudaLaunchKernel[cooperative]
   * 2. Barrier Wait. No CUDA call is permitted
   * 3. Enqueue Events. CUDA event wait/enqueue.
   * This is needed because step 2 cannot call any CUDA primitive, otherwise if
   * cudaFree happens between 1 and 3, it could block that CUDA call and
   * prevent some ranks from launching their network threads, which would
   * prevent the NCCL call from completing, blocking the cudaFree call.
   */

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ncclBarrierEnqueue(comm), ret, end);

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  NCCLCHECKGOTO(ncclBarrierEnqueueWait(comm), ret, end);

  if (comm->userStream == NULL) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
  }
  NCCLCHECKGOTO(ncclEnqueueEvents(comm), ret, end);

end:
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling cudaSetDevice, because this call can fail too
  return ret;
}
