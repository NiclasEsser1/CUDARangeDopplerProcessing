#include "CudaKernels.cuh"
#include "CudaBase.cuh"
#include <cuda_runtime_api.h>
#include <math.h>


__global__ void windowHamming(float* idata, int length)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (tidx < length)
	{
		idata[tidx] = 0.54 - 0.46 * cos(2*tidx*PI_F / (length - 1));
	}
}
__global__ void windowHann(float* idata, int length)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (tidx < length)
	{
		idata[tidx] = 0.5*(1 + cos(2*tidx*PI_F / (length - 1)));
	}
}
__global__ void windowBartlett(float* idata, int length)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (tidx < length)
	{
		idata[tidx] = 0;
	}
}
__global__ void windowBlackman(float* idata, int length)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (tidx < length)
	{
		idata[tidx] = 0.74 / 2 * -0.5 * cos(2 * PI_F*tidx / (length - 1)) + 0.16 / 2 * sin(4 * PI_F*tidx / (length - 1));
	}
}

__global__ void windowMultiplyReal(float* idata, float* window, int length, int height, int depth)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tidz = threadIdx.z + blockIdx.z*blockDim.z;
	if(tidx < length && tidy < height && tidz < depth)
	{
		idata[tidx*tidy] = window[tidx] * idata[tidx*tidy];
	}
}


__global__ void windowMultiplyCplx(cufftComplex* idata, float* window, int length, int height, int depth)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tidz = threadIdx.z + blockIdx.z*blockDim.z;
	if(tidx < length && tidy < height && tidz < depth)
	{
		idata[tidx*tidy].x = window[tidx] * idata[tidx*tidy].x;
		idata[tidx*tidy].y = window[tidx] * idata[tidx*tidy].y;
	}
}

__global__ void transposeBufferGlobalReal(float* idata, float* odata)
{
	int x = blockIdx.x * TRANSPOSE_DIM + threadIdx.x;
	int y = blockIdx.y * TRANSPOSE_DIM+ threadIdx.y;
	int width = gridDim.x * TRANSPOSE_DIM;
	// if(x < length && y < height)
	// {
		for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
		{
			odata[x*width + (y+j)] = idata[(y+j)*width + x];
		}
	// }
}

__global__ void transposeBufferGlobalCplx(cufftComplex* idata, cufftComplex* odata)
{
	int x = blockIdx.x * TRANSPOSE_DIM + threadIdx.x;
	int y = blockIdx.y * TRANSPOSE_DIM+ threadIdx.y;
	int width = gridDim.x * TRANSPOSE_DIM;
	// if(x < length && y < height)
	// {
		for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
		{
			odata[x*width + (y+j)] = idata[(y+j)*width + x];
		}
	// }
}

__global__ void transposeBufferSharedReal(float* idata, float* odata)
{
	__shared__ float tile[TRANSPOSE_DIM][TRANSPOSE_DIM];

	int x = blockIdx.x * TRANSPOSE_DIM + threadIdx.x;
	int y = blockIdx.y * TRANSPOSE_DIM + threadIdx.y;
	int width = gridDim.x * TRANSPOSE_DIM;

	for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
	{
		tile[threadIdx.y+j][threadIdx.x] = (idata[(y+j)*width + x]);
	}

	__syncthreads();

	x = blockIdx.y * TRANSPOSE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TRANSPOSE_DIM + threadIdx.y;
	for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
	{
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}

}

__global__ void transposeBufferSharedCplx(cufftComplex* idata, cufftComplex* odata)
{
	__shared__ cufftComplex tile[TRANSPOSE_DIM][TRANSPOSE_DIM];

	int x = blockIdx.x * TRANSPOSE_DIM + threadIdx.x;
	int y = blockIdx.y * TRANSPOSE_DIM + threadIdx.y;
	int width = gridDim.x * TRANSPOSE_DIM;

	for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
	{
		tile[threadIdx.y+j][threadIdx.x] = (idata[(y+j)*width + x]);
	}

	__syncthreads();

	x = blockIdx.y * TRANSPOSE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TRANSPOSE_DIM + threadIdx.y;
	for (int j = 0; j < TRANSPOSE_DIM; j+= TRANSPOSE_ROWS)
	{
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}

}
