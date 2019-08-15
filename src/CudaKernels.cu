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

__global__ void windowMultiplyReal(float* idata, float* window, int width, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if(tidx < width && tidy < height)
	{
		idata[tidy * width + tidx] = window[tidx] * idata[tidy * width + tidx];
	}
}


__global__ void windowMultiplyCplx(cufftComplex* idata, float* window, int width, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if(tidx < width && tidy < height)
	{
		idata[tidy * width + tidx].x = window[tidx] * idata[tidy * width + tidx].x;
		idata[tidy * width + tidx].y = window[tidx] * idata[tidy * width + tidx].y;
	}
}

__global__ void transposeBufferGlobalReal(float* idata, float* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y+ threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidx*height + tidy] = idata[tidy*width + tidx];
	}
}

__global__ void transposeBufferGlobalCplx(cufftComplex* idata, cufftComplex* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidx*height + tidy].x = idata[tidy*width + tidx].x;
		odata[tidx*height + tidy].y = idata[tidy*width + tidx].y;
	}
}

__global__ void absoluteKernel(cufftComplex* idata, float* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidy*width + tidx] = sqrt(idata[tidy*width + tidx].x * idata[tidy*width + tidx].x + idata[tidy*width + tidx].y*idata[tidy*width + tidx].y);
	}
}

__global__ void colormapJet(float* idata, unsigned char* odata, int max, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char c_max = 0x00;
	unsigned char c_min = 0x00;
	if(tidx < width && tidy < height)
	{
		c_max = (unsigned char)127*idata[tidx + width*tidy]/max;
		c_min = ~c_max;
		odata[(tidx + width*tidy) * 3 + 0] = c_max;
		odata[(tidx + width*tidy) * 3 + 1] = (unsigned char)63*idata[tidx + width*tidy]/max;
		odata[(tidx + width*tidy) * 3 + 2] = c_min;
	}
}

__global__ void colormapHot(float* idata, unsigned char* odata, int max, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char c_max = 0x00;
	unsigned char c_min = 0x00;
	if(tidx < width && tidy < height)
	{
		c_max = (unsigned char)127*idata[tidx + width*tidy]/max;
		c_min = ~c_max;
		odata[(tidx + width*tidy) * 3 + 0] = c_max;
		odata[(tidx + width*tidy) * 3 + 1] = (unsigned char)63*idata[tidx + width*tidy]/max;
		odata[(tidx + width*tidy) * 3 + 2] = c_min;
	}
}

__global__ void colormapCold(float* idata, unsigned char* odata, int max, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if(tidx < width && tidy < height)
	{
		odata[tidx + width*tidy] = (unsigned char)255*idata[tidx + width*tidy]/max;
	}
}

__global__ void colormapBlue(float* idata, unsigned char* odata, int max, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if(tidx < width && tidy < height)
	{
		odata[tidx + width*tidy] = (unsigned char)255*idata[tidx + width*tidy]/max;
	}
}
