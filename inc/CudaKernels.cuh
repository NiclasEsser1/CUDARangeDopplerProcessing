#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

#include <cuda_runtime_api.h>

#include "CudaBase.cuh"
#include "device_launch_parameters.h"
#define PI_F   3.14159f

__global__ void windowHamming(float* idata, int length);
__global__ void windowHann(float* idata, int length);
__global__ void windowBartlett(float* idata, int length);
__global__ void windowBlackman(float* idata, int length);
__global__ void windowMultiplyReal(float* idata, float* window, int length, int height = 1, int depth = 1);
__global__ void windowMultiplyCplx(cufftComplex* idata, float* window, int length, int height = 1, int depth = 1);
__global__ void zeroPaddingCplx(cufftComplex* idata, int pad_length, int length, int height = 1, int depth = 1);
__global__ void transposeBufferGlobalReal(float* idata, float* odata);
__global__ void transposeBufferGlobalCplx(cufftComplex* idata, cufftComplex* odata);
__global__ void transposeBufferSharedReal(float* idata, float* odata);
__global__ void transposeBufferSharedCplx(cufftComplex* idata, cufftComplex* odata);

#endif
