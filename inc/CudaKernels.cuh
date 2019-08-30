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

__global__ void windowKernel(float* idata, float* window, int width, int height);
__global__ void windowKernel(cufftComplex* idata, float* window, int width, int height);

__global__ void transposeGlobalKernel(float* idata, float* odata, int width, int height);
__global__ void transposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height);
__global__ void hermetianTransposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height);

__global__ void transposeSharedKernel(float* idata, float* odata, int height);
__global__ void transposeSharedKernel(cufftComplex* idata, cufftComplex* odata, int height);

__global__ void absoluteKernel(cufftComplex* idata, float* odata, int width, int height);
//template <typename T>__global__ void getMaxValueKernel(T* idata, int width, int height);

__global__ void colormapJet(float* idata, unsigned char* odata, float max, float min, int width, int height);
__global__ void colormapViridis(float* idata, unsigned char* odata, float max, float min, int width, int height);
__global__ void colormapAccent(float* idata, unsigned char* odata, float max, float min, int width, int height);
__global__ void colormapMagma(float* idata, unsigned char* odata, float max, float min, int width, int height);
__global__ void colormapInferno(float* idata, unsigned char* odata, float max, float min, int width, int height);
__global__ void colormapBlue(float* idata, unsigned char* odata, float max, float min, int width, int height);

template <typename T>__global__ void maxKernel(T* idata, int count);
// template <typename T>__global__ void maxKernel(T* idata, int width, int height);
template <typename T>__global__ void minKernel(T* idata, int count);




#endif
