#include "cudabase.cuh"
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

void CudaBase::setWindow(float* idata, int win_len, int type, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "Calculating window tabs" << endl;
#endif
	if(height == 0)
	{
	 	int tx = MAX_NOF_THREADS;
		int bx = win_len / MAX_NOF_THREADS + 1;
		dim3 blockSize(tx);
		dim3 gridSize(bx);
		switch (type)
		{
			case HAMMING:
				windowHamming <<<gridSize, blockSize >>> (idata, win_len);
				break;
			case HANN:
				windowHann <<<gridSize, blockSize >>> (idata, win_len);
				break;
			case BARTLETT:
				windowBartlett <<<gridSize, blockSize >>> (idata, win_len);
				break;
			case BLACKMAN:
				windowBlackman <<<gridSize, blockSize >>> (idata, win_len);
				break;
		}
	}
	else
	{
	 	int tx = MAX_NOF_THREADS;
		int bx = win_len/tx + 1;
		int by = height;
		dim3 blockSize(tx);
		dim3 gridSize(bx, by);
		switch (type)
		{
			case HAMMING:
				windowHamming2d<<<gridSize, blockSize >>> (idata, win_len, height);
				break;
			case HANN:
				windowHann2d<<<gridSize, blockSize >>> (idata, win_len, height);
				break;
			case BARTLETT:
				windowBartlett2d <<<gridSize, blockSize >>> (idata, win_len, height);
				break;
			case BLACKMAN:
				windowBlackman2d <<<gridSize, blockSize >>> (idata, win_len, height);
				break;
		}
	}
}

template <typename T>
void CudaBase::window(T* idata, float* window, int width, int height, int dim)
{

#ifdef PRINT_KERNEL_LAUNCH
	cout << "windowing..." << endl;
#endif
	int tx = MAX_NOF_THREADS;
	int bx = width/tx+1;
	int by = height;

	dim3 blockSize(tx);
	dim3 gridSize(bx, by);
	if(dim == 1)
		windowKernel<<<gridSize,blockSize>>>(idata, window, width, height);
	else if (dim == 2)
		window2dKernel<<<gridSize,blockSize>>>(idata, window, width, height);
	// CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}
template void CudaBase::window<float>(float*, float*, int, int, int);
template void CudaBase::window<cufftComplex>(cufftComplex*, float*, int, int, int);

void CudaBase::convert(short* idata, float* odata, int count)
{

#ifdef PRINT_KERNEL_LAUNCH
	cout << "converting float to short..." << endl;
#endif
	int tx = MAX_NOF_THREADS;
	int bx = count / MAX_NOF_THREADS + 1;

	dim3 blockSize(tx);
	dim3 gridSize(bx);
	convertKernel<<<gridSize,blockSize>>>(idata, odata, count);
}


/*
*	Mathematical opertaions
*/
void CudaBase::absolute(cufftComplex* idata, float* odata, int width, int height)
{

#ifdef PRINT_KERNEL_LAUNCH
	cout << "calculatin absolute values of complex numbers..." << endl;
#endif
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

#ifdef PRINT_KERNEL_LAUNCH
	cout << "hermitian transposing... ";
#endif
	int tx = TILE_DIM;
	int ty = TILE_DIM;
	int bx = width/tx;
	int by = height/ty;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	// If matrix is a square matrix
	if(width == height)
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "(transpose shared)" << endl;
#endif
		ty = BLOCK_ROWS;
		bx = width/TILE_DIM;
		by = height/TILE_DIM;
		dim3 blockSize(tx,ty);
		dim3 gridSize(bx,by);
		hermetianTransposeSharedKernel<<<gridSize, blockSize>>>(odata);
		return;
	}
	if(idata == NULL)
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "(temporary buffer)" << endl;
#endif
		CudaVector<cufftComplex>* temp = new CudaVector<cufftComplex>(device, width*height);
		CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), odata, temp->getSize(), cudaMemcpyDeviceToDevice));
		hermetianTransposeGlobalKernel<<<gridSize, blockSize>>>(temp->getDevPtr(), odata, width, height);
		temp->freeMemory();
		//delete(temp);
	}
	else
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "(two buffers)" << endl;
#endif
		hermetianTransposeGlobalKernel<<<gridSize, blockSize>>>(idata, odata, width, height);
	}
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}

void CudaBase::hermetianTransposeShared(cufftComplex* data, int width, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "hermitian transposing (shared memory)... ";
#endif
	int tx = TILE_DIM;
	int ty = BLOCK_ROWS;
	int bx = width/TILE_DIM;
	int by = height/TILE_DIM;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	hermetianTransposeSharedKernel<<<gridSize, blockSize>>>(data);
	//CUDA_CHECK(cudaDeviceSynchronize());
}


template <typename T>
void CudaBase::transpose(T* odata, int width, int height, T* idata)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "transposing... ";
#endif
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	//printf("Transposing buffer... ");
	if(idata == NULL)
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "(temporary buffer)" << endl;
#endif
		CudaVector<T>* temp = new CudaVector<T>(device, width*height);
		CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), odata, temp->getSize(), cudaMemcpyDeviceToDevice));
		transposeGlobalKernel<<<gridSize, blockSize>>>(temp->getDevPtr(), odata, width, height);
		temp->freeMemory();
		//delete(temp);
	}
	else
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "(two buffers)" << endl;
#endif
		transposeGlobalKernel<<<gridSize, blockSize>>>(idata, odata, width, height);
	}
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("done\n");
}
template void CudaBase::transpose<float>(float*, int, int, float*);
template void CudaBase::transpose<cufftComplex>(cufftComplex*, int, int, cufftComplex*);


template <typename T>
void CudaBase::transposeShared(T* data, int width, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "hermitian transposing (shared memory)... "  << endl;
#endif
	int tx = TILE_DIM;
	int ty = BLOCK_ROWS;
	int bx = width/TILE_DIM;
	int by = height/TILE_DIM;
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	transposeSharedKernel<<<gridSize, blockSize>>>(data);
	//CUDA_CHECK(cudaDeviceSynchronize());
}
template void CudaBase::transposeShared<float>(float*,int,int);
// template void CudaBase::transposeShared<cufftComplex>(cufftComplex*, int, int, cufftComplex*);


void CudaBase::fftshift(cufftComplex* data, int n, int batch)
{
	int tx = MAX_NOF_THREADS;
	int bx = n/tx+1;
	int by = batch;
	dim3 blockSize(tx);
	dim3 gridSize(bx,by);
	if(batch > 1)
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "fftshift 2d... "  << endl;
#endif
		fftshift2d<<<gridSize, blockSize>>>(data, n, batch);
	}
	else
	{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "fftshift 1d... "  << endl;
#endif
		fftshift1d<<<gridSize, blockSize>>>(data, n);
	}
}


template <typename T>
T CudaBase::max(T* idata, int width, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "max values (slow))... "  << endl;
#endif
	int count = width*height;
	int tx = MAX_NOF_THREADS;
	int bx = count/tx;
	float max_val = 0;
	dim3 blockSize(tx);
	dim3 gridSize(bx);

	CudaVector<T>* temp = new CudaVector<T>(device, count);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	maxKernel<T><<<gridSize, blockSize>>>(temp->getDevPtr(), count);
	//CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&max_val, temp->getDevPtr(0), sizeof(T), cudaMemcpyDeviceToHost));
	temp->freeMemory();
	//delete(temp);
	return max_val;
}
template float CudaBase::max<float>(float*, int, int);
template int CudaBase::max<int>(int*, int, int);
template char CudaBase::max<char>(char*, int, int);
template double CudaBase::max<double>(double*, int, int);


template <typename T>
T CudaBase::min(T* idata, int width, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "min values (slow))... "  << endl;
#endif
	int count = width*height;
	int tx = MAX_NOF_THREADS;
	int bx = count/tx;
	float min_val = 0;
	dim3 blockSize(tx);
	dim3 gridSize(bx);

	CudaVector<T>* temp = new CudaVector<T>(device, count);
	CUDA_CHECK(cudaMemcpy(temp->getDevPtr(), idata, temp->getSize(), cudaMemcpyDeviceToDevice));

	minKernel<T><<<gridSize, blockSize>>>(temp->getDevPtr(), count);
	//CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&min_val, temp->getDevPtr(0), sizeof(T), cudaMemcpyDeviceToHost));
	temp->freeMemory();
	//delete(temp);
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
#ifdef PRINT_KERNEL_LAUNCH
	cout << "max & min values (fast)... "  << endl;
#endif
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
void CudaBase::initJpegEncoder(int width, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "init jpg encoder... "  << endl;
#endif
	CUDA_CHECK(cudaStreamCreate(&jpeg_stream));
	jpeg_encoder = gpujpeg_encoder_create(&jpeg_stream);

	//
	gpujpeg_set_default_parameters(&jpeg_param);
	jpeg_param.quality = 80;
	jpeg_param.restart_interval = 16;
	jpeg_param.interleaved = 1;

	gpujpeg_image_set_default_parameters(&jpeg_param_image);
	jpeg_param_image.width = width;
	jpeg_param_image.height = height;
	jpeg_param_image.comp_count = 3;
	jpeg_param_image.color_space = GPUJPEG_RGB;
	jpeg_param_image.pixel_format = GPUJPEG_444_U8_P012;

	// Use default sampling factors
	gpujpeg_parameters_chroma_subsampling_422(&jpeg_param);
	if ( jpeg_encoder == NULL )
		jpeg_encoder = gpujpeg_encoder_create(&jpeg_stream);

}
void CudaBase::encodeBmpToJpeg(unsigned char* idata, uint8_t* odata, int* p_jpeg_size, string path)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "encode bmp -> jpg... "  << endl;
#endif
	static int image_num = 0;
	uint8_t* jpeg_image = NULL;
	gpujpeg_encoder_input_set_image(&jpeg_encoder_input, idata);
	gpujpeg_encoder_encode(jpeg_encoder, &jpeg_param, &jpeg_param_image, &jpeg_encoder_input, &jpeg_image, p_jpeg_size);
	CUDA_CHECK(cudaStreamSynchronize(jpeg_stream));

	// odata = (uint8_t*) malloc(*p_jpeg_size);
	if(odata != nullptr)
		memcpy((void*)odata, (void*)jpeg_image, *p_jpeg_size);
	else
		printf("Could not allocated memory for jpeg jpeg_image\n");
	if(path != "")
		gpujpeg_image_save_to_file((path+".jpg").c_str(), (uint8_t*)jpeg_image, *p_jpeg_size);
	//CUDA_CHECK(cudaStreamDestroy(jpeg_stream));
	image_num++;
}

void CudaBase::destroyJpegEncoder()
{
	CUDA_CHECK(cudaStreamDestroy(jpeg_stream));
	gpujpeg_image_destroy(jpeg_image);
	gpujpeg_encoder_destroy(jpeg_encoder);
}

void CudaBase::random(cufftComplex* idata, int cnt, float mean, float stddev)
{
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	if(mean == 0)
	{

		CURAND_CALL(curandGenerateUniform(gen, (float*)&idata[0].x, cnt));
		CURAND_CALL(curandGenerateUniform(gen, (float*)&idata[0].y, cnt));
	}
	else
	{
		CURAND_CALL(curandGenerateNormal(gen, (float*)&idata[0].x, cnt,mean, stddev));
		CURAND_CALL(curandGenerateNormal(gen, (float*)&idata[0].y, cnt,mean, stddev));
	}
	CURAND_CALL(curandDestroyGenerator(gen));
}

void CudaBase::random(float* idata, int cnt, float mean, float stddev)
{
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	if(mean == 0)
	{
		CURAND_CALL(curandGenerateUniform(gen, idata, cnt));
	}
	else
	{
		CURAND_CALL(curandGenerateNormal(gen, idata, cnt,mean, stddev));
	}
	CURAND_CALL(curandDestroyGenerator(gen));
}

void CudaBase::random(unsigned int* idata, int cnt)
{
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	CURAND_CALL(curandGenerate(gen, idata, cnt));
	CURAND_CALL(curandDestroyGenerator(gen));
}

void CudaBase::mapColors(float* idata, unsigned char* odata, int width, int height, int type, int scale)
{
	int tx = 32;
	int ty = 32;
	int bx = width/tx+1;
	int by = height/ty+1;

	float max;
	float min;
#ifdef PRINT_KERNEL_LAUNCH
	cout << "map colors... " << endl;
#endif
	CudaVector<float>* temp = new CudaVector<float>(device, width*height);
	minMax(idata, temp->getDevPtr(), width*height);
	CUDA_CHECK(cudaMemcpy(&min, temp->getDevPtr(0), sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&max, temp->getDevPtr(1), sizeof(float), cudaMemcpyDeviceToHost));
	temp->freeMemory();
	//delete(temp);
	dim3 blockSize(tx,ty);
	dim3 gridSize(bx,by);
	switch(type)
	{
		case JET:
			colormapJet<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		case VIRIDIS:
			colormapViridis<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		case ACCENT:
			colormapAccent<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		case MAGMA:
			colormapMagma<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		case INFERNO:
			colormapInferno<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		case BLUE:
			colormapBlue<<<gridSize,blockSize>>>(idata, odata, max, min, width, height, scale);
			break;
		default:
			cout << "This colormapping is not provided" << endl;

	}
}


/*
*	FFT functions
*/

void CudaBase::r2c1dFFT(cufftComplex* odata, int n, int batch, cufftReal* idata)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "Real -> complex FFT (1d)... "  << endl;
#endif
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
#ifdef PRINT_KERNEL_LAUNCH
	cout << "complex -> complex IFFT (1d)... "  << endl;
#endif
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
#ifdef PRINT_KERNEL_LAUNCH
	cout << "complex -> complex FFT (1d)... "  << endl;
#endif
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
#ifdef PRINT_KERNEL_LAUNCH
	cout << "hilbertransform... "  << endl;
#endif
	//printf("Performing hilbert transform... \n");
	r2c1dFFT(odata, n, batch, idata);
	c2c1dInverseFFT(odata, n/2+1, batch);
	//printf("done\n");
}

template <typename T>
void CudaBase::zeroFilling(T* idata, int row, int length, int height)
{
#ifdef PRINT_KERNEL_LAUNCH
	cout << "zero filling... "  << endl;
#endif
	int tx = MAX_NOF_THREADS;
	int bx = tx/length+1;
	int by = height;
	dim3 blockSize(tx);
	dim3 gridSize(bx,by);
	zeroFillingKernel<<<gridSize,blockSize>>>(idata, row, length, height);
}
template void CudaBase::zeroFilling<float>(float*, int, int, int);
template void CudaBase::zeroFilling<cufftComplex>(cufftComplex*, int, int, int);

void CudaBase::printWindowTaps(float* idata, int win_len)
{
	float* help = (float*)malloc(win_len*sizeof(float));
	CUDA_CHECK(cudaMemcpy(help, idata, win_len*sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < win_len; i++)
		printf("Tap[%d] = %f\n",i,help[i]);
}
