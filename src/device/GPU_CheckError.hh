//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   device/GPU_CheckError.hh
 * \author Kelly (KT) Thompson
 * \brief  Provide helper macros for CUDA code
 * \note   Copyright (C) 2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef device_GPU_CheckError_hh
#define device_GPU_CheckError_hh

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace rtt_device {

inline void CheckError(cudaError_t const err, char const *const fun,
                       const int line) {
  if (err) {
    printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n", err,
           cudaGetErrorString(err), fun, line);
    exit(1);
  }
}

inline void CheckErrorMsg(cudaError_t const err, char const *const msg,
                          char const *const fun, int const line) {
  if (err) {
    printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n%s\n", err,
           cudaGetErrorString(err), fun, line, msg);
    exit(1);
  }
}

} // end namespace rtt_device

//----------------------------------------------------------------------------//
// Helper functions for CUDA runtime error checking
//----------------------------------------------------------------------------//

#define DBS_CHECK_ERROR(err)                                                   \
  rtt_device::CheckError(err, __FUNCTION__, __LINE__);
#define DBS_CHECK_ERRORMSG(err, msg)                                           \
  rtt_device::CheckErrorMsg(err, msg, __FUNCTION__, __LINE__);

#endif // device_GPU_CheckError_hh

//---------------------------------------------------------------------------//
// end of device/GPU_CheckError.hh
//---------------------------------------------------------------------------//
