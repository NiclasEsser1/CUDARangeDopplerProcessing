#include "CudaBase.cuh"

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



void CudaBase::setWindow(float* idata, int win_len, winType type)
{
	int tx = MAX_NOF_THREADS;
	int bx = win_len / MAX_NOF_THREADS + 1;

	dim3 blockSize(tx);
	dim3 gridSize(bx);

	switch (type)
	{
		case HAMMING:
			//printf("Calculate hamming window... ");
			windowHamming <<<gridSize, blockSize >>> (idata, win_len);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case HANN:
			//printf("Calculate hann window... ");
			windowHann <<<gridSize, blockSize >>> (idata, win_len);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BARTLETT:
			//printf("Calculate bartlett window... ");
			windowBartlett <<<gridSize, blockSize >>> (idata, win_len);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLACKMAN:
			//printf("Calculate blackman window... ");
			windowBlackman <<<gridSize, blockSize >>> (idata, win_len);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
	}
	//printf("done!\n");
}

template <typename T>
void CudaBase::window(T* idata, float* window, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);
	//printf("Performing windowing (real)... ");
	windowKernel<<<gridSize,blockSize>>>(idata, window, width, height);
	// CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}
template void CudaBase::window<float>(float*, float*, int, int);
template void CudaBase::window<cufftComplex>(cufftComplex*, float*, int, int);

void CudaBase::absolute(cufftComplex* idata, float* odata, int width, int height)
{
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);
	//printf("Calculating absolute values... ");
	absoluteKernel<<<gridSize,blockSize>>>(idata, odata, width, height);
	// CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::hermitianTranspose(cufftComplex* odata, int width, int height, cufftComplex* idata)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	if(idata == NULL)
	{
		CudaVector<cufftComplex>* temp = new CudaVector<cufftComplex>(device, width*height);
		CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), odata, temp->getSize(), cudaMemcpyDeviceToDevice));
		hermetianTransposeGlobalKernel<<<gridSize, blockSize>>>(temp->getDevPtr(), odata, width, height);
		temp->resize(0);
		delete(temp);
	}
	else
	{
		hermetianTransposeGlobalKernel<<<gridSize, blockSize>>>(idata, odata, width, height);
	}
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::fftshift(cufftComplex* data, int n, int batch)
{
	int tx = MAX_NOF_THREADS;
	int bx = tx/n+1;
	int by = batch;
	dim3 blockSize(tx);
	dim3 gridSize(bx,by);
	if(batch > 1)
	{
		fftshift2d<<<gridSize, blockSize>>>(data, n, batch);
	}
	else
	{
		fftshift1d<<<gridSize, blockSize>>>(data, n);
	}
}

void CudaBase::encodeBmpToJpeg(unsigned char* idata, unsigned char* odata, int width, int height)
{
	nvjpegHandle_t nv_handle;
	nvjpegEncoderState_t nv_enc_state;
	nvjpegEncoderParams_t nv_enc_params;
	cudaStream_t stream;
	nvjpegImage_t source;

	source.channel[0] = idata;
	source.pitch[0] = width*3;

	CUDA_CHECK(cudaStreamCreate(&stream));

	// initialize nvjpeg structures
	CUDA_CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
	CUDA_CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
	CUDA_CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
	//CUDA_CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));

	// Compress image
	CUDA_CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &source, NVJPEG_INPUT_RGB, width, height, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaDeviceSynchronize());

	// get compressed stream size
	size_t length;
	CUDA_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));


	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, odata, &length, 0));
	printf("Length is: %ld \n", length);
	CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void CudaBase::transpose(T* odata, int width, int height, T* idata)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	if(idata == NULL)
	{
		CudaVector<T>* temp = new CudaVector<T>(device, width*height);
		CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), odata, temp->getSize(), cudaMemcpyDeviceToDevice));
		transposeGlobalKernel<<<gridSize, blockSize>>>(temp->getDevPtr(), odata, width, height);
		temp->resize(0);
		delete(temp);
	}
	else
	{
		transposeGlobalKernel<<<gridSize, blockSize>>>(idata, odata, width, height);
	}
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}
template void CudaBase::transpose<float>(float*, int, int, float*);
template void CudaBase::transpose<cufftComplex>(cufftComplex*, int, int, cufftComplex*);

template <typename T>
void CudaBase::transposeShared(T* odata, int width, int height, T* idata)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	if(idata == NULL)
	{
		CudaVector<T>* temp = new CudaVector<T>(device, width*height);
		CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), odata, temp->getSize(), cudaMemcpyDeviceToDevice));
		transposeSharedKernel<<<gridSize, blockSize>>>(temp->getDevPtr(), odata, height);
		temp->resize(0);
		delete(temp);
	}
	else
	{
		transposeSharedKernel<<<gridSize, blockSize>>>(idata, odata, height);
	}
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}
template void CudaBase::transposeShared<float>(float*, int, int, float*);
template void CudaBase::transposeShared<cufftComplex>(cufftComplex*, int, int, cufftComplex*);


template <typename T>
T CudaBase::max(T* idata, int width, int height)
{
	int count = width*height;
	int tx = MAX_NOF_THREADS;
	int bx = count/tx;
	float max_val = 0;
	dim3 blockSize(tx);
	dim3 gridSize(bx);

	CudaVector<T>* temp = new CudaVector<T>(device, count);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	maxKernel<T><<<gridSize, blockSize>>>(temp->getDevPtr(), count);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&max_val, temp->getDevPtr(0), sizeof(T), cudaMemcpyDeviceToHost));
	temp->resize(0);
	delete(temp);
	return max_val;
}
template float CudaBase::max<float>(float*, int, int);
template int CudaBase::max<int>(int*, int, int);
template char CudaBase::max<char>(char*, int, int);
template double CudaBase::max<double>(double*, int, int);



template <typename T>
T CudaBase::min(T* idata, int width, int height)
{
	int count = width*height;
	int tx = MAX_NOF_THREADS;
	int bx = count/tx;
	float min_val = 0;
	dim3 blockSize(tx);
	dim3 gridSize(bx);

	CudaVector<T>* temp = new CudaVector<T>(device, count);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	minKernel<T><<<gridSize, blockSize>>>(temp->getDevPtr(), count);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&min_val, temp->getDevPtr(0), sizeof(T), cudaMemcpyDeviceToHost));
	temp->resize(0);
	delete(temp);
	return min_val;
}
template float CudaBase::min<float>(float*, int, int);
template int CudaBase::min<int>(int*, int, int);
template char CudaBase::min<char>(char*, int, int);
template double CudaBase::min<double>(double*, int, int);



void CudaBase::mapColors(float* idata, unsigned char* odata, int width, int height, color_t type)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	CUDA_CHECK(cudaDeviceSynchronize());
	float max_v = max(idata, width, height);
	float min_v = min(idata, width, height);
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);

	switch(type)
	{
		case JET:
			colormapJet<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case VIRIDIS:
			colormapViridis<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
		case ACCENT:
			colormapAccent<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
		case MAGMA:
			colormapMagma<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case INFERNO:
			colormapInferno<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
			break;
		case BLUE:
			colormapBlue<<<gridSize,blockSize>>>(idata, odata, max_v, min_v, width, height);
			// CUDA_CHECK(cudaDeviceSynchronize());
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
	r2c1dFFT(odata, n, batch, idata);
	c2c1dInverseFFT(odata, n/2+1, batch);
	//printf("done\n");
}



void CudaBase::printWindowTaps(float* idata, int win_len)
{
	float* help = (float*)malloc(win_len*sizeof(float));
	CUDA_CHECK(cudaMemcpy(help, idata, win_len*sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < win_len; i++)
		printf("Tap[%d] = %f\n",i,help[i]);
}
