#include "CudaBase.cuh"
//#include <libgpujpeg/gpujpeg.h>
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

/*
* 	Windowing
*/

void CudaBase::setWindow(float* idata, int win_len, int type)
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

/*
*	Mathematical opertaions
*/
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
	int tx = TILE_DIM;
	int ty = TILE_DIM;
	int bx = width/tx;
	int by = height/ty;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	// If matrix is a square matrix
	if(width == height)
	{
		hermetianTransposeSharedKernel<<<gridSize, blockSize>>>(odata);
	}
	else if(idata == NULL)
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
	CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}


template <typename T> void CudaBase::transpose(T* odata, int width, int height, T* idata)
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


template <typename T> void CudaBase::transposeShared(T* data, int width, int height)
{
	int tx = TILE_DIM;
	int ty = BLOCK_ROWS;
	int bx = width/TILE_DIM;
	int by = height/TILE_DIM;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	transposeSharedKernel<<<gridSize, blockSize>>>(data);
	CUDA_CHECK(cudaDeviceSynchronize());
}
template void CudaBase::transposeShared<float>(float*,int,int);
// template void CudaBase::transposeShared<cufftComplex>(cufftComplex*, int, int, cufftComplex*);


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


void CudaBase::findBlockSize(int* whichSize, int* num_el)
{
	const float pretty_big_number = 24.0f*1024.0f*1024.0f;
	float ratio = float((*num_el))/pretty_big_number;
	if(ratio > 0.8f)
		(*whichSize) =  5;
	else if(ratio > 0.6f)
		(*whichSize) =  4;
	else if(ratio > 0.4f)
		(*whichSize) =  3;
	else if(ratio > 0.2f)
		(*whichSize) =  2;
	else
		(*whichSize) =  1;
}


void CudaBase::minMax(float* idata, float* odata, int num_els)
{

	int whichSize = -1;
	findBlockSize(&whichSize,&num_els);

	int block_size = powf(2,whichSize-1)*blockSize1;
	int num_blocks = num_els/block_size;
	int tail = num_els - num_blocks*block_size;
	int start_adr = num_els - tail;

	if(whichSize == 1)
		find_min_max<blockSize1,threads><<< num_blocks, threads>>>(idata, odata);
	else if(whichSize == 2)
		find_min_max<blockSize1*2,threads><<< num_blocks, threads>>>(idata, odata);
	else if(whichSize == 3)
		find_min_max<blockSize1*4,threads><<< num_blocks, threads>>>(idata, odata);
	else if(whichSize == 4)
		find_min_max<blockSize1*8,threads><<< num_blocks, threads>>>(idata, odata);
	else
		find_min_max<blockSize1*16,threads><<< num_blocks, threads>>>(idata, odata);
	find_min_max_dynamic<threads><<<1, threads>>>(idata, odata, num_els, start_adr, num_blocks);
}
/*
*	Image processing
*/
void CudaBase::encodeBmpToJpeg(unsigned char* idata, uint8_t* odata, int* p_jpeg_size, int width, int height)
{
	static int image_num = -1;
	char image_dir[256];
	cudaStream_t stream;
	uint8_t* image = NULL;


	CUDA_CHECK(cudaStreamCreate(&stream));
	image_num++;
	struct gpujpeg_encoder_input encoder_input;
	struct gpujpeg_parameters param;
	struct gpujpeg_image_parameters param_image;
	struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(&stream);

	sprintf(image_dir, "/home/niclas/SoftwareProjekte/Cuda/PerformanceComparsion/results/img/streaming/jpeg/CH%d_%d.jpg", image_num%4, image_num/4);
	gpujpeg_set_default_parameters(&param);
	param.quality = 80;
	param.restart_interval = 16;
	param.interleaved = 1;

	gpujpeg_image_set_default_parameters(&param_image);
	param_image.width = width;
	param_image.height = height;
	param_image.comp_count = 3;
	param_image.color_space = GPUJPEG_RGB;
	param_image.pixel_format = GPUJPEG_444_U8_P012;

	// Use default sampling factors
	gpujpeg_parameters_chroma_subsampling_422(&param);
	if ( encoder == NULL )
		encoder = gpujpeg_encoder_create(&stream);

	gpujpeg_encoder_input_set_image(&encoder_input, idata);
	gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image, p_jpeg_size);
	gpujpeg_image_save_to_file(image_dir, image, *p_jpeg_size);
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// odata = (uint8_t*) malloc(*p_jpeg_size);
	if(odata != nullptr)
		memcpy((void*)odata, (void*)image, *p_jpeg_size);
	else
		printf("Could not allocated memory for jpeg image\n");

	CUDA_CHECK(cudaStreamDestroy(stream));
	gpujpeg_image_destroy(image);
	gpujpeg_encoder_destroy(encoder);
}


void CudaBase::mapColors(float* idata, unsigned char* odata, int width, int height, color_t type)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;

	float max_v;
	float min_v;

	CudaVector<float>* temp = new CudaVector<float>(device, width*height);
	minMax(idata, temp->getDevPtr(), width*height);
	cudaMemcpy(&min_v, temp->getDevPtr(0), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_v, temp->getDevPtr(1), sizeof(float), cudaMemcpyDeviceToHost);
	temp->resize(0);
	delete(temp);
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
