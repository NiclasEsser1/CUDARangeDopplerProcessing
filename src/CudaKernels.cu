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

__global__ void windowKernel(float* idata, float* window, int width, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if(tidx < width && tidy < height)
	{
		idata[tidy * width + tidx] = window[tidx] * idata[tidy * width + tidx];
	}
}


__global__ void windowKernel(cufftComplex* idata, float* window, int width, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if(tidx < width && tidy < height)
	{
		idata[tidy * width + tidx].x = window[tidx] * idata[tidy * width + tidx].x;
		idata[tidy * width + tidx].y = window[tidx] * idata[tidy * width + tidx].y;
	}
}

__global__ void transposeGlobalKernel(float* idata, float* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y+ threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidx*height + tidy] = idata[tidy*width + tidx];
	}
}

__global__ void transposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidx*height + tidy].x = idata[tidy*width + tidx].x;
		odata[tidx*height + tidy].y = idata[tidy*width + tidx].y;
	}
}

__global__ void transposeSharedKernel(float* idata, float* odata, int height)
{
	__shared__ float tile[32][32];

	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int width = gridDim.x * 32;

	for (int j = 0; j < 32; j += height/32)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

	__syncthreads();

	x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
	y = blockIdx.x * 32 + threadIdx.y;

	for (int j = 0; j < 32; j += height/32)
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeSharedKernel(cufftComplex* idata, cufftComplex* odata, int height)
{
	__shared__ cufftComplex tile[32][32];

	  int x = blockIdx.x * 32 + threadIdx.x;
	  int y = blockIdx.y * 32 + threadIdx.y;
	  int width = gridDim.x * 32;

	  for (int j = 0; j < 32; j += height/32)
	  {
		  tile[threadIdx.y+j][threadIdx.x].x = idata[(y+j)*width + x].x;
		  tile[threadIdx.y+j][threadIdx.x].y = idata[(y+j)*width + x].y;
	  }

	  __syncthreads();

	  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
	  y = blockIdx.x * 32 + threadIdx.y;

	  for (int j = 0; j < 32; j += height/32)
	  {
		  odata[(y+j)*width + x].x = tile[threadIdx.x][threadIdx.y + j].x;
		  odata[(y+j)*width + x].y = tile[threadIdx.x][threadIdx.y + j].y;
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
	int colormap_index = (int)idata[tidx + width*tidy]/max*(JET_SIZE-2);
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = colormap_jet[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = colormap_jet[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = colormap_jet[colormap_index][2];
	}
}

__global__ void colormapHot(float* idata, unsigned char* odata, int max, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = 0;
		odata[(tidx + width*tidy) * 3 + 1] = 255*idata[tidx + width*tidy]/max;
		odata[(tidx + width*tidy) * 3 + 2] = 0;
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
