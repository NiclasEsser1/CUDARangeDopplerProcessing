#include "CudaBase.cuh"
#include "CudaGPU.cuh"
#include "CudaKernels.cuh"
#include <cuda_runtime_api.h>
#include <cufft.h>


/**
_________
PUBLIC
_________
**/
CudaBase::CudaBase(CudaGPU* device, int length, int height, int depth)
{
	setDevice(device);
	setLength(length);
	setHeight(height);
	setDepth(depth);
	setBytes((int)sizeof(float));
	printf("Needed memory capacity: %lf of %lu (MByte)\n", (float)x*y*z*bytes*12/(1024*1024), (long unsigned)device->getProperties().totalGlobalMem/(1024*1024));
	if(x*y*z*bytes*12 < (long unsigned)device->getProperties().totalGlobalMem)
	{
		initDeviceEnv();
	}
	else
	{
		printf("Specfied dimensions exceeds gloabal memory (%lu kByte) of device %d \n", (long unsigned)device->getProperties().totalGlobalMem/1024, (int)device->getDeviceID());
	}
}

CudaBase::~CudaBase()
{
	// freeCudaVector(input);
	// freeCudaVector(windowBuf);
	// freeCudaVector(filterBuf);
	// freeCudaVector(fftBuf);
	// freeCudaVector(rngDopBuf);
	// freeCudaVector(output);
	// freeDeviceMem(*input->getDevPtr());
	// freeDeviceMem(*windowBuf->getDevPtr());
	// freeDeviceMem(*filterBuf->getDevPtr());
	// freeDeviceMem(*fftBuf->getDevPtr());
	// freeDeviceMem(*rngDopBuf->getDevPtr());
	// freeDeviceMem(*output->getDevPtr());
}

void CudaBase::rangeDopplerProcessing()
{
	hilbertTransform(input->getDevPtr(), fftBuf->getDevPtr());
	window(fftBuf->getDevPtr());
	doFFT(fftBuf->getDevPtr());
	doFFT(fftBuf->getDevPtr(), rngDopBuf->getDevPtr(), true);
}

void CudaBase::hilbertTransform(float* idata, cufftComplex* odata)
{
	int nfft[1] = {x};
	int rank = 1;
	printf("Performing hilbert transform...\n");
	r2cManyFFT(idata, odata, nfft, rank);
	c2cManyInPlaceIFFT(odata, nfft, rank);
	printf("done\n");
}

template<typename T> void CudaBase::window(T* idata)
{
	int tx = MAX_NOF_THREADS;
	int bx = 1;
	int by = y;
	int bz = z;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by, bz);

	if(x > tx)
	{
		bx = x/tx+0.5;
	}
	if(win_kind == REAL)
	{
		printf("Performing windowing (real)...\n");
		windowMultiplyReal<<<gridSize,blockSize>>>((float*)idata, windowBuf->getDevPtr(), y, z);
	}
	else if(win_kind == COMPLEX)
	{
		printf("Performing windowing (complex)...\n");
		windowMultiplyCplx<<<gridSize,blockSize>>>((cufftComplex*)idata, windowBuf->getDevPtr(), y, z);
	}
	printf("done\n");
}
void CudaBase::windowCplx(cufftComplex* idata)
{
	int tx = MAX_NOF_THREADS;
	int bx = 1;
	int by = y;
	int bz = z;
	if(x > tx)
	{
		bx = x/tx+0.5;
	}

	dim3 blockSize(tx);
	dim3 gridSize(bx, by, bz);
	windowMultiplyCplx<<<gridSize,blockSize>>>((cufftComplex*)idata, windowBuf->getDevPtr(), y, z);
}

void CudaBase::doFFT(cufftComplex* idata, cufftComplex* odata, bool transpose)
{
	int nfft[1] = {x};
	int rank = 1;
	if(transpose = true && odata != NULL)
	{
		printf("Transposing input matrix...\n");
		dim3 blockSize(TRANSPOSE_DIM,TRANSPOSE_ROWS);
		dim3 gridSize(TRANSPOSE_DIM,TRANSPOSE_DIM);
		transposeBufferGlobalCplx<<<gridSize, blockSize>>>(idata, odata);
		printf("done\n");
		printf("Performing %dD-FFT\n", rank);
		c2cManyInPlaceFFT(odata, nfft, rank);
		printf("done\n");

	}
	else
	{
		if(transpose == true)
			printf("Insert an output buffer for transposing matrix\n");
		printf("Performing %dD-FFT\n", rank);
		c2cManyInPlaceFFT(idata, nfft, rank);
		printf("done\n");
	}
}

void CudaBase::make_medianfilter()
{

}


void CudaBase::setLength(int val)
{
	x = val;
}

void CudaBase::setHeight(int val)
{
	y = val;
}

void CudaBase::setDepth(int val)
{
	z = val;
}

void CudaBase::setBytes(int val)
{
	bytes = val;
}

void CudaBase::setMedFilter(int val)
{
	if(val < 10)
	{
		m_filter.x = val;
	}
	else
	{
		printf("Median Filter has a max size of 9x9\n");
	}
}

void CudaBase::setDevice(CudaGPU* val)
{
	device = val;
}

void CudaBase::setProcessingBuffer(float* idata, cudaMemcpyKind kind)
{
	printf("Copying signal from host to device...\n");
	memCopyBuffer((void*)input->getDevPtr(), (void*)idata, x*y*z*sizeof(*idata), cudaMemcpyHostToDevice);
	printf("done!\n");
}

void CudaBase::setWindow(winType type, numKind kind, winLocation loc)
{
	win_type = type;
	win_kind = kind;
	win_loc = loc;
	win_len = x;
	calculateWindowTaps();
}

/**
_________
PROTECTED
_________
**/

template<typename T> void CudaBase::allocateDeviceMem(T* buf, size_t elements)
{
	CUDA_CHECK(cudaMalloc((void**)&buf, elements*sizeof(T)));
}


template<typename T> void CudaBase::memCopyBuffer(T* dst, T* src, size_t size, cudaMemcpyKind kind)
{
	CUDA_CHECK(cudaMemcpy((void*)dst, (void*)src, size, kind));
}

void CudaBase::freeDeviceMem(void* ptr)
{
	if (ptr != NULL)
	{
		CUDA_CHECK(cudaFree(ptr));
	}
}

template<typename T>  void CudaBase::freeCudaVector(CudaVector<T*>* vec)
{
	if (vec != NULL)
	{
		vec->CudaVector();
	}
}

void CudaBase::initDeviceEnv()
{
	//Allocate device memory for processing chain
	printf("Init processing envirioment\n");
	input = new CudaVector<float>(x*y*z*sizeof(*input));
	windowBuf = new CudaVector<float>(x*sizeof(*windowBuf));
	fftBuf = new CudaVector<cufftComplex>((x*y*z / 2 + 1)*sizeof(*fftBuf));
	rngDopBuf = new CudaVector<cufftComplex>((x*y*z / 2 + 1)*sizeof(*rngDopBuf));
	filterBuf = new CudaVector<float>(m_filter.x*m_filter.x*sizeof(*filterBuf));
	output = new CudaVector<float>( (x*y*z / 2 + 1) *sizeof(*output));
	printf("Allocated device memory with success\n");
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

void CudaBase::c2cManyInPlaceFFT(cufftComplex *data, int *nfft, int  rank)
{
	cufftHandle plan;
	int length = nfft[0];
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlanMany(
		&plan, rank, nfft,
		nfft, length, rank,
		nfft, length, rank,
		CUFFT_C2C, length
	));
	// Execute in place FFT (destination = source)
	CUDA_CHECK_FFT(cufftExecC2C(plan, data, data, CUFFT_FORWARD));
	CUDA_CHECK_FFT(cufftDestroy(plan));
}

void CudaBase::c2cManyInPlaceIFFT(cufftComplex *data, int *nfft, int  rank)
{
	cufftHandle plan;
	int length = nfft[0];
	// Plan for FFT
	CUDA_CHECK_FFT(cufftPlanMany(
		&plan, rank, nfft,
		nfft, length, rank,
		nfft, length, rank,
		CUFFT_C2C, length
	));
	// Execute in place FFT (destination = source)
	CUDA_CHECK_FFT(cufftExecC2C(plan, data, data, CUFFT_INVERSE));
	CUDA_CHECK_FFT(cufftDestroy(plan));
}

void CudaBase::calculateWindowTaps()
{
	int blocks = 1;
	if (win_len > MAX_NOF_THREADS)
		blocks = win_len / MAX_NOF_THREADS + 0.5;

	dim3 blockSize(MAX_NOF_THREADS);
	dim3 gridSize(blocks);
	switch (win_loc)
	{
	case HOST:
		switch (win_type)
		{
		case HAMMING:
			break;
		case HANN:
			break;
		case BARTLETT:
			break;
		case BLACKMAN:
			break;
		}
		break;
	case D_GLOBAL:
		switch (win_type)
		{
		case HAMMING:
			printf("Calculate hamming window...\n");
			windowHamming <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HANN:
			printf("Calculate hann window...\n");
			windowHann <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BARTLETT:
			printf("Calculate bartlett window...\n");
			windowBartlett <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLACKMAN:
			printf("Calculate blackman window...\n");
			windowBlackman <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		}
		break;
	case D_SHARED:
		switch (win_type)
		{
		case HAMMING:
			printf("Calculate hamming window...\n");
			windowHamming <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HANN:
			printf("Calculate hann window...\n");
			windowHann << <gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BARTLETT:
			printf("Calculate bartlett window...\n");
			windowBartlett <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLACKMAN:
			printf("Calculate blackman window...\n");
			windowBlackman <<<gridSize, blockSize >>> (windowBuf->getDevPtr(), win_len);
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		}
		break;
	}
	printf("done\n");
}

void CudaBase::printWindowTaps()
{
	float* help = (float*)malloc(win_len*sizeof(float));
	memCopyBuffer(help, windowBuf->getDevPtr(), win_len*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < win_len; i++)
		printf("Tap[%d] = %f\n",i,help[i]);
}
