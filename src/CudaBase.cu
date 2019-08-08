#include "CudaBase.cuh"
#include "CudaGPU.cuh"
#include "CudaKernels.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <unistd.h>
#include <iostream>

/**
_________
PUBLIC
_________
**/
CudaBase::CudaBase(CudaGPU* device)
{
	setDevice(device);
	setLength(0);
	setHeight(0);
}

CudaBase::~CudaBase()
{
	freeMemory();
}

void CudaBase::freeMemory()
{
	freeCudaVector(floatBuffer);
	freeCudaVector(windowBuf);
	freeCudaVector(complexBuffer);
	freeCudaVector(transposeBuf);
}

bool CudaBase::initDeviceEnv()
{
	//Allocate device memory for processing chain
	printf("Allocate memory for processing buffer\n");
	total_used_mem = x_size * y_size * 3  + x_size;
	printf("needed mem: %ld and total avaible mem: %ld\n", total_used_mem/(1024*1024),device->totalMemory()/(1024*1024));
	if(total_used_mem < device->totalMemory())
	{
		floatBuffer = new CudaVector<float>(device, x_size * y_size);
		complexBuffer = new CudaVector<cufftComplex>(device, x_size * y_size);
		transposeBuf = new CudaVector<cufftComplex>(device, x_size * y_size);
		windowBuf = new CudaVector<float>(device, x_size);
		return 1;
	}
	else
	{
		printf("Not enoguh memory avaible on the used device, aborting... \n");
		return 0;
	}
}

void CudaBase::setWindow(winType type, numKind kind)
{
	win_type = type;
	win_kind = kind;
	win_len = x_size;
	calculateWindowTaps();
}

void CudaBase::windowReal(float* idata, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);

	//printf("Performing windowing (real)... ");
	windowMultiplyReal<<<gridSize,blockSize>>>((float*)idata, windowBuf->getDevPtr(), width, height);
	//printf("done\n");
}

void CudaBase::windowCplx(cufftComplex* idata, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);

	//printf("Performing windowing (complex)... ");
	windowMultiplyCplx<<<gridSize,blockSize>>>(idata, windowBuf->getDevPtr(), width, height);
	//printf("done\n");
}

void CudaBase::calculateWindowTaps()
{
	int tx = MAX_NOF_THREADS;
	int bx = win_len / MAX_NOF_THREADS + 1;

	dim3 blockSize(tx);
	dim3 gridSize(bx);

	switch (win_type)
	{
		case HAMMING:
			//printf("Calculate hamming window... ");
			windowHamming <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HANN:
			//printf("Calculate hann window... ");
			windowHann <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BARTLETT:
			//printf("Calculate bartlett window... ");
			windowBartlett <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLACKMAN:
			//printf("Calculate blackman window... ");
			windowBlackman <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
	}
	//printf("done!\n");
}

void CudaBase::absolute(cufftComplex* idata, float* odata, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);
	//printf("Calculating absolute values... ");
	absoluteKernel<<<gridSize,blockSize>>>(idata, odata, width, height);
	//printf("done\n");
}

void CudaBase::transpose(cufftComplex* idata, cufftComplex* odata, int width, int height)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	transposeBufferGlobalCplx<<<gridSize, blockSize>>>(idata, odata, width, height);
	//printf("done\n");
}

void CudaBase::r2c1dFFT(float* idata, cufftComplex *odata, int n, int batch)
{
	//printf("Performing 1D FFT (r2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_R2C, batch));
	CUDA_CHECK_FFT(cufftExecR2C(plan, (cufftReal*)idata, odata));
	CUDA_CHECK_FFT(cufftDestroy(plan));
	//printf("done! \n");
}

void CudaBase::c2c1dIFFT(cufftComplex* idata, int n, int batch)
{
	//printf("Performing 1D inverse FFT (c2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
	CUDA_CHECK_FFT(cufftExecC2C(plan, idata, idata, CUFFT_INVERSE));
	CUDA_CHECK_FFT(cufftDestroy(plan));
	//printf("done! \n");
}

void CudaBase::c2c1dFFT(cufftComplex* idata, int n, int batch)
{
	//printf("Performing 1D FFT (c2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
	// Execute in place FFT (destination = source)
	CUDA_CHECK_FFT(cufftExecC2C(plan, idata, idata, CUFFT_FORWARD));
	CUDA_CHECK_FFT(cufftDestroy(plan));
	//printf("done! \n");
}

void CudaBase::r2cManyFFT(float* idata, cufftComplex *odata, int *nfft, int  rank)
{
	cufftHandle plan;
	int length = nfft[0];
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlanMany(
		&plan, rank, nfft,
		nfft, length, rank,
		nfft, length, rank,
		CUFFT_R2C, length
	));
	// Execute in place FFT (destination = source)
	CUDA_CHECK_FFT(cufftExecR2C(plan, idata, odata));
	CUDA_CHECK_FFT(cufftDestroy(plan));
}

void CudaBase::hilbertTransform(float* idata, cufftComplex* odata, int n, int batch)
{
	//printf("Performing hilbert transform... \n");
	r2c1dFFT(idata, odata, n, batch);
	c2c1dIFFT(odata, n/2+1, batch);
	//printf("done\n");
}

void CudaBase::printWindowTaps()
{
	float* help = (float*)malloc(win_len*sizeof(float));
	CUDA_CHECK(cudaMemcpy(help, windowBuf->getDevPtr(), win_len*sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < win_len; i++)
		printf("Tap[%d] = %f\n",i,help[i]);
}
