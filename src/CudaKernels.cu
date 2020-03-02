#include "CudaKernels.cuh"



__global__ void windowHamming(float* idata, int length)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	if (tidx < length)
	{
	printf("tidx:%d", tidx);
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

__global__ void windowHamming2d(float* idata, int length, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	//printf("tidy: %d, tidy:%d, idx:%d", tidy,tidx ,tidy * length + tidx);
	if (tidx < length && tidy < height)
	{
		//printf("tidy: %d, tidy:%d, idx:%d", tidy,tidx ,tidy * length + tidx);
		idata[tidy * length + tidx] = (0.54 - 0.46 * cos(2*tidy*PI_F / (height - 1))) * (0.54 - 0.46 * cos(2*tidx*PI_F / (length - 1)));
	}
}

__global__ void windowHann2d(float* idata, int length, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (tidx < length && tidy < height)
	{
		idata[tidy * length + tidx] =  0.5*(1 + cos(2*tidy*PI_F / (height - 1))) * 0.5*(1 + cos(2*tidx*PI_F / (length - 1)));
	}
}
__global__ void windowBartlett2d(float* idata, int length, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (tidx < length && tidy < height)
	{
		idata[tidy * length + tidx] = 0;
	}
}
__global__ void windowBlackman2d(float* idata, int length, int height)
{
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (tidx < length && tidy < height)
	{
		idata[tidy * length + tidx] = (0.74 / 2 * -0.5 * cos(2 * PI_F*tidy / (height - 1)) + 0.16 / 2 * sin(4 * PI_F*tidy / (height - 1)))
			* (0.74 / 2 * -0.5 * cos(2 * PI_F*tidx / (length - 1)) + 0.16 / 2 * sin(4 * PI_F*tidx / (length - 1)));
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


__global__ void transposeSharedKernel(float* data)
{
	__shared__ float tile_s[TILE_DIM][TILE_DIM+1];
    __shared__ float tile_d[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
	// handle off-diagonal case
    if (blockIdx.y>blockIdx.x)
	{
      int dx = blockIdx.y * TILE_DIM + threadIdx.x;
      int dy = blockIdx.x * TILE_DIM + threadIdx.y;
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_d[threadIdx.y+j][threadIdx.x] = data[(dy+j)*width + dx];
      __syncthreads();
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(dy+j)*width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(y+j)*width + x] = tile_d[threadIdx.x][threadIdx.y + j];
    }
	// handle on-diagonal case
    else if (blockIdx.y==blockIdx.x)
	{
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
      __syncthreads();
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(y+j)*width + x] = tile_s[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void hermetianTransposeGlobalKernel(cufftComplex* idata, cufftComplex* odata, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx < width && tidy < height)
	{
		odata[tidx*height + tidy].x = idata[tidy*width + tidx].x;
		odata[tidx*height + tidy].y = (-1)*idata[tidy*width + tidx].y;
	}
}

__global__ void hermetianTransposeSharedKernel(cufftComplex* data)
{
	__shared__ cufftComplex tile_s[TILE_DIM][TILE_DIM+1];
    __shared__ cufftComplex tile_d[TILE_DIM][TILE_DIM+1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	// handle off-diagonal case
	if (blockIdx.y>blockIdx.x)
	{
		int dx = blockIdx.y * TILE_DIM + threadIdx.x;
		int dy = blockIdx.x * TILE_DIM + threadIdx.y;
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile_s[threadIdx.y+j][threadIdx.x].x = data[(y+j)*width + x].x;
			tile_s[threadIdx.y+j][threadIdx.x].y = (-1)*data[(y+j)*width + x].y;
		}
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile_d[threadIdx.y+j][threadIdx.x].x = data[(dy+j)*width + dx].x;
			tile_d[threadIdx.y+j][threadIdx.x].y = (-1)*data[(dy+j)*width + dx].y;
		}
		__syncthreads();
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			data[(dy+j)*width + dx].x = tile_s[threadIdx.x][threadIdx.y + j].x;
			data[(dy+j)*width + dx].y = tile_s[threadIdx.x][threadIdx.y + j].y;
		}
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			data[(y+j)*width + x].x = tile_d[threadIdx.x][threadIdx.y + j].x;
			data[(y+j)*width + x].y = tile_d[threadIdx.x][threadIdx.y + j].y;
		}
	}

	// handle on-diagonal case
	else if (blockIdx.y==blockIdx.x)
	{
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile_s[threadIdx.y+j][threadIdx.x].x = data[(y+j)*width + x].x;
			tile_s[threadIdx.y+j][threadIdx.x].y = (-1)*data[(y+j)*width + x].y;
		}
		__syncthreads();
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			data[(y+j)*width + x].x = tile_s[threadIdx.x][threadIdx.y + j].x;
			data[(y+j)*width + x].y = tile_s[threadIdx.x][threadIdx.y + j].y;
		}
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



__global__ void colormapJet(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)((idata[tidx + width*tidy]-min)/(max-min)*(JET_SIZE-1));

	if(tidx < width && tidy < height)
	{
		odata[(tidx + width * tidy) * 3 + 0] = (unsigned char)255*colormap_jet[colormap_index][0];
		odata[(tidx + width * tidy) * 3 + 1] = (unsigned char)255*colormap_jet[colormap_index][1];
		odata[(tidx + width * tidy) * 3 + 2] = (unsigned char)255*colormap_jet[colormap_index][2];
	}
}

__global__ void colormapViridis(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)(idata[tidx + width*tidy]-min)/(max-min)*(VIRIDIS_SIZE-1);
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = (unsigned char)255*colormap_viridis[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = (unsigned char)255*colormap_viridis[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = (unsigned char)255*colormap_viridis[colormap_index][2];
	}
}


__global__ void colormapAccent(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)(idata[tidx + width*tidy]-min)/(max-min)*(ACCENT_SIZE-1);
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = colormap_accent[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = colormap_accent[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = colormap_accent[colormap_index][2];
	}
}

__global__ void colormapMagma(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)((idata[tidx + width*tidy]-min)/(max-min)*(MAGMA_SIZE-1));
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = (unsigned char)255*colormap_magma[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = (unsigned char)255*colormap_magma[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = (unsigned char)255*colormap_magma[colormap_index][2];
	}
}

__global__ void colormapInferno(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)(idata[tidx + width*tidy]-min)/(max-min)*(INFERNO_SIZE-1);
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = (unsigned char)255*colormap_inferno[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = (unsigned char)255*colormap_inferno[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = (unsigned char)255*colormap_inferno[colormap_index][2];
	}
}

__global__ void colormapBlue(float* idata, unsigned char* odata, float max, float min, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int colormap_index = (int)(idata[tidx + width*tidy]-min)/(max-min)*(BLUE_SIZE-1);
	if(tidx < width && tidy < height)
	{
		odata[(tidx + width*tidy) * 3 + 0] = colormap_blue[colormap_index][0];
		odata[(tidx + width*tidy) * 3 + 1] = colormap_blue[colormap_index][1];
		odata[(tidx + width*tidy) * 3 + 2] = colormap_blue[colormap_index][2];
	}
}

template <typename T>__global__ void maxKernel(T* idata, int count)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = count/2;

	if(tidx < i)
	{
		while(i != 0)
		{
			if(idata[tidx] < idata[tidx+i])
				idata[tidx] = idata[tidx+i];
			__syncthreads();
			if(idata[0] < idata[tidx])
				idata[0] = idata[tidx];
			__syncthreads();
			i /= 2;
		}
	}
}
template __global__ void maxKernel<float>(float*, int);
template __global__ void maxKernel<int>(int*, int);
template __global__ void maxKernel<char>(char*, int);
template __global__ void maxKernel<double>(double*, int);

template <typename T>__global__ void minKernel(T* idata, int count)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = count/2;

	if(tidx < i)
	{
		while(i != 0)
		{
			if(idata[tidx] > idata[tidx+i])
				idata[tidx] = idata[tidx+i];
			__syncthreads();
			if(idata[0] > idata[tidx])
				idata[0] = idata[tidx+i];
			__syncthreads();
			i /= 2;
		}
	}
}
template __global__ void minKernel<float>(float*, int);
template __global__ void minKernel<int>(int*, int);
template __global__ void minKernel<char>(char*, int);
template __global__ void minKernel<double>(double*, int);

template <typename T>
__global__ void fftshift1d(T* data, int n)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidx < n/2)
	{
		// Save the first value
        T regTemp = data[tidx];

        // Swap the first element
        data[tidx] = (T) data[tidx + (n / 2)];

        // Swap the second one
        data[tidx + (n / 2)] = (T) regTemp;
	}
}
template __global__ void fftshift1d<cufftComplex>(cufftComplex*, int);
template __global__ void fftshift1d<float>(float*, int);

template <typename T>
__global__ void fftshift2d(T* data, int n, int batch)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y;

	if(tidx < n/2 && tidy < batch)
	{
		// Save the first value
        T regTemp = data[tidx + tidy * n];

        // Swap the first element
        data[tidx + tidy * n] = (T) data[tidx + tidy * n + (n / 2)];

        // Swap the second one
        data[tidx + tidy * n + (n / 2)] = (T) regTemp;
	}
}
template __global__ void fftshift2d<cufftComplex>(cufftComplex*, int, int);
template __global__ void fftshift2d<float>(float*, int, int);


__device__ void warp_reduce_max( float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ?
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ?
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ?
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ?
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ?
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ?
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}


__device__ void warp_reduce_min( float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ?
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ?
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ?
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ?
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ?
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ?
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}

template<int threads> __global__ void find_min_max_dynamic(float* in, float* out, int n, int start_adr, int num_blocks)
{
	__shared__ float smem_min[64];
	__shared__ float smem_max[64];
	int tid = threadIdx.x + start_adr;
	float max = -inf;
	float min = inf;
	float val;
	// tail part
	int mult = 0;
	for(int i = 1; mult + tid < n; i++)
	{
		val = in[tid + mult];

		min = val < min ? val : min;
		max = val > max ? val : max;

		mult = int_mult(i,threads);
	}
	// previously reduced MIN part
	mult = 0;
	int i;
	for(i = 1; mult+threadIdx.x < num_blocks; i++)
	{
		val = out[threadIdx.x + mult];

		min = val < min ? val : min;

		mult = int_mult(i,threads);
	}
	// MAX part
	for(; mult+threadIdx.x < num_blocks*2; i++)
	{
		val = out[threadIdx.x + mult];

		max = val > max ? val : max;

		mult = int_mult(i,threads);
	}
	if(threads == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;

	}
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;
	__syncthreads();
	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}
}
template __global__ void find_min_max_dynamic<64>(float*, float*, int, int, int);
//template __global__ void find_min_max_dynamic<int>(float*, float*, int, int, int);

template<int els_per_block, int threads> __global__ void find_min_max(float* in, float* out)
{
	__shared__ float smem_min[64];
	__shared__ float smem_max[64];
	int tid = threadIdx.x + blockIdx.x*els_per_block;
	float max = -inf;
	float min = inf;
	float val;

	const int iters = els_per_block/threads;
#pragma unroll
		for(int i = 0; i < iters; i++)
		{

			val = in[tid + i*threads];

			min = val < min ? val : min;
			max = val > max ? val : max;

		}
	if(threads == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;

	}
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;
	__syncthreads();
	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}
}
template __global__ void find_min_max<2048, 64>(float*, float*);
template __global__ void find_min_max<4096, 64>(float*, float*);
template __global__ void find_min_max<8192, 64>(float*, float*);
template __global__ void find_min_max<16384, 64>(float*, float*);
template __global__ void find_min_max<32768, 64>(float*, float*);
