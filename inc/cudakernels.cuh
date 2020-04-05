#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

#include "colormaps.cuh"
#include "cudagpu.cuh"

#include <math.h>
//#include "device_launch_parameters.h"
#define PI_F   3.14159f
#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif
#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)
#else
#define int_mult(x,y)	x*y
#endif
#define inf 0x7f800000

#define TILE_DIM 32			// for transpose functions
#define BLOCK_ROWS 8		// for transpose functions

__global__ void windowHamming(float* idata, int length);
__global__ void windowHann(float* idata, int length);
__global__ void windowBartlett(float* idata, int length);
__global__ void windowBlackman(float* idata, int length);
__global__ void windowHamming2d(float* idata, int length, int height);
__global__ void windowHann2d(float* idata, int length, int height);
__global__ void windowBartlett2d(float* idata, int length, int height);
__global__ void windowBlackman2d(float* idata, int length, int height);

__global__ void windowKernel(float* idata, float* window, int width, int height);
__global__ void windowKernel(cufftComplex* idata, float* window, int width, int height);
__global__ void window2dKernel(float* idata, float* window, int width, int height);
__global__ void window2dKernel(cufftComplex* idata, float* window, int width, int height);
__global__ void convertKernel(short* idata, float* odata, int size);

__global__ void transposeGlobalKernel(float* idata, float* odata, int width, int height);
__global__ void transposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height);
__global__ void transposeSharedKernel(float* data);
// __global__ void transposeSharedKernel(cufftComplex* data);

__global__ void hermetianTransposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height);
__global__ void hermetianTransposeSharedKernel(cufftComplex* data);

__global__ void absoluteKernel(cufftComplex* idata, float* odata, int width, int height);

__global__ void colormapJet(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void colormapViridis(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void colormapAccent(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void colormapMagma(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void colormapInferno(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void colormapBlue(float* idata, unsigned char* odata, float max, float min, int width, int height, int scale = LOG);
__global__ void zeroFillingKernel(float* idata, int row, int length, int height);
__global__ void zeroFillingKernel(cufftComplex* idata, int row, int length, int height);

template <typename T>__global__ void maxKernel(T* idata, int count);
template <typename T>__global__ void minKernel(T* idata, int count);

template <typename T>__global__ void fftshift2d(T* data, int n, int batch);
template <typename T>__global__ void fftshift1d(T* data, int n);

__device__ void warp_reduce_max( float smem[64]);
__device__ void warp_reduce_min( float smem[64]);
template<int threads> __global__ void find_min_max_dynamic(float* in, float* out, int n, int start_adr, int num_blocks);
template<int els_per_block, int threads> __global__ void find_min_max(float* in, float* out);

#endif
