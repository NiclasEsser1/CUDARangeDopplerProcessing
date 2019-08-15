#include "CudaBase.cuh"
#include "CudaGPU.cuh"
#include "CudaKernels.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdlib.h>
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
}

CudaBase::~CudaBase()
{

}



void CudaBase::setWindow(float* idata, int width, winType type, numKind kind)
{
	win_type = type;
	win_kind = kind;
	win_len = width;
	calculateWindowTaps(idata);
}

void CudaBase::windowReal(float* idata, float* window, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);

	//printf("Performing windowing (real)... ");
	windowMultiplyReal<<<gridSize,blockSize>>>((float*)idata, window, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::windowCplx(cufftComplex* idata, float* window, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);

	//printf("Performing windowing (complex)... ");
	windowMultiplyCplx<<<gridSize,blockSize>>>(idata, window, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::calculateWindowTaps(float* idata)
{
	int tx = MAX_NOF_THREADS;
	int bx = win_len / MAX_NOF_THREADS + 1;

	dim3 blockSize(tx);
	dim3 gridSize(bx);

	switch (win_type)
	{
		case HAMMING:
			//printf("Calculate hamming window... ");
			windowHamming <<<gridSize, blockSize >>> (idata, win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HANN:
			//printf("Calculate hann window... ");
			windowHann <<<gridSize, blockSize >>> (idata, win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BARTLETT:
			//printf("Calculate bartlett window... ");
			windowBartlett <<<gridSize, blockSize >>> (idata, win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLACKMAN:
			//printf("Calculate blackman window... ");
			windowBlackman <<<gridSize, blockSize >>> (idata, win_len);
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
	CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::transpose(cufftComplex* idata, int width, int height)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	CudaVector<cufftComplex>* temp = new CudaVector<cufftComplex>(device, width*height);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	transposeBufferGlobalCplx<<<gridSize, blockSize>>>(temp->getDevPtr(), idata, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());
	delete(temp);
	//printf("done\n");
}
template <typename T>
T CudaBase::getMaxValue(T* idata, int width, int height)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	float max = 0;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);

	CudaVector<T>* temp = new CudaVector<T>(device, width*height);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	getMaxValueKernel<T><<<gridSize, blockSize>>>(temp->getDevPtr(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&max, &temp->getDevPtr()[0], sizeof(float), cudaMemcpyDeviceToHost));
	delete(temp);
	return max;
}

void CudaBase::renderImage(float* idata, unsigned char* odata, int width, int height, color_t type)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	float max = 0;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	max = getMaxValue(idata, width, height);
	switch(type)
	{
		case JET:
			colormapJet<<<gridSize,blockSize>>>(idata, odata, width, height, max);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HOT:
			colormapHot<<<gridSize,blockSize>>>(idata, odata, width, height, max);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case COLD:
			colormapCold<<<gridSize,blockSize>>>(idata, odata, width, height, max);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLUE:
			colormapBlue<<<gridSize,blockSize>>>(idata, odata, width, height, max);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
	}
}
/*
*	FFT functions
*/

void CudaBase::r2c1dFFT(cufftComplex* odata, int n, int batch, cufftReal* idata)
{
	//printf("Performing 1D FFT (r2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_R2C, batch));
	if(idata == NULL)
	{
		CUDA_CHECK_FFT(cufftExecR2C(plan, (cufftReal*)odata, odata));
	}
	else
	{
		CUDA_CHECK_FFT(cufftExecR2C(plan, idata, odata));
	}
	CUDA_CHECK_FFT(cufftDestroy(plan));
	//printf("done! \n");
}

void CudaBase::c2c1dInverseFFT(cufftComplex* idata, int n, int batch)
{
	//printf("Performing 1D inverse FFT (c2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
	CUDA_CHECK(cudaDeviceSynchronize());
	//CUDA_CHECK_FFT(cufftExecC2C(plan, idata, idata, CUFFT_INVERSE));
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK_FFT(cufftDestroy(plan));
	CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done! \n");
}

void CudaBase::c2c1dFFT(cufftComplex* idata, int n, int batch)
{
	//printf("Performing 1D FFT (c2c)... ");
	cufftHandle plan;
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
	CUDA_CHECK(cudaDeviceSynchronize());
	// Execute in place FFT (destination = source)
	CUDA_CHECK_FFT(cufftExecC2C(plan, idata, idata, CUFFT_FORWARD));
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK_FFT(cufftDestroy(plan));
	CUDA_CHECK(cudaDeviceSynchronize());
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
	r2c1dFFT(odata, n, batch, idata);
	c2c1dInverseFFT(odata, n/2+1, batch);
	//printf("done\n");
}



void CudaBase::printWindowTaps(float* idata)
{
	float* help = (float*)malloc(win_len*sizeof(float));
	CUDA_CHECK(cudaMemcpy(help, idata, win_len*sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < win_len; i++)
		printf("Tap[%d] = %f\n",i,help[i]);
}
