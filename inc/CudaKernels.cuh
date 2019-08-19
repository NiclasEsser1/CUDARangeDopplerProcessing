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

__global__ void transposeSharedKernel(float* idata, float* odata, int height);
__global__ void transposeSharedKernel(cufftComplex* idata, cufftComplex* odata, int height);

__global__ void absoluteKernel(cufftComplex* idata, float* odata, int width, int height);
//template <typename T>__global__ void getMaxValueKernel(T* idata, int width, int height);
__global__ void colormapJet(float* idata, unsigned char* odata, int max, int width, int height);
__global__ void colormapHot(float* idata, unsigned char* odata, int max, int width, int height);
__global__ void colormapCold(float* idata, unsigned char* odata, int max, int width, int height);
__global__ void colormapBlue(float* idata, unsigned char* odata, int max, int width, int height);


template <typename T>__global__ void maxKernel(T* idata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = width/2;

	if(tidx < width/2 && tidy < height)
	{
		while(i != 0)
		{
			if(tidx < i)
				if(idata[tidx + width * tidy] < idata[tidx+i + width * tidy])
					idata[tidx + width * tidy] = idata[tidx+i + width * tidy];
			// __syncthreads();
			i /= 2;
		}
		i = height/2;
		while(i != 0)
		{
			if(tidy < i)
				if(idata[width * tidy] < idata[(i + tidy)*width])
					idata[width * tidy] = idata[(i + tidy)*width];
			// __syncthreads();
			i /= 2;
		}
	}
}

template <typename T>__global__ void minKernel(T* idata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = width/2;

	if(tidx < width/2 && tidy < height)
	{
		while(i != 0)
		{
			if(tidx < i)
				if(idata[tidx + width * tidy] > idata[tidx+i + width * tidy])
					idata[tidx + width * tidy] = idata[tidx+i + width * tidy];
			// __syncthreads();
			i /= 2;
		}
		i = height/2;
		while(i != 0)
		{
			if(tidy < i)
				if(idata[width * tidy] > idata[(i + tidy)*width])
					idata[width * tidy] = idata[(i + tidy)*width];
			// __syncthreads();
			i /= 2;
		}
	}
}

#define JET_SIZE 9
#define ACCENT_SIZE 8

__device__ unsigned char colormap_jet[JET_SIZE][3] = {
     {0x00, 0x00, 0x7F},
     {0x00, 0x00, 0xFF},
     {0x00, 0x7F, 0xFF},
     {0x00, 0xFF, 0xFF},
     {0x7F, 0xFF, 0x7F},
     {0xFF, 0xFF, 0x00},
     {0xFF, 0x7F, 0x00},
     {0xFF, 0xFF, 0x00},
     {0x7F, 0x00, 0x00}
};

#endif
