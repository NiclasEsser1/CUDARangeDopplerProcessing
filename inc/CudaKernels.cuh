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
__global__ void windowMultiplyReal(float* idata, float* window, int width, int height);
__global__ void windowMultiplyCplx(cufftComplex* idata, float* window, int width, int height);
__global__ void transposeBufferGlobalReal(float* idata, float* odata, int width, int height);
__global__ void transposeBufferGlobalCplx(cufftComplex* idata, cufftComplex* odata, int width, int height);
__global__ void absoluteKernel(cufftComplex* idata, float* odata, int width, int height);
#endif
