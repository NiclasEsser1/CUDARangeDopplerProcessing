#include "CudaAlgorithm.cuh"

CudaAlgorithm::CudaAlgorithm(CudaBase* obj_base, int width, int height, int depth, int c_depth)
{
    base = obj_base;
    device = base->getDevice();
	x_size = width;
    y_size = height;
	z_size = depth;
    color_depth = c_depth;
	allocated = false;
    charBuffer = NULL;
	floatBuffer = NULL;
	windowBuffer = NULL;
	complexBuffer = NULL;
}

CudaAlgorithm::~CudaAlgorithm()
{
	freeMemory();
}

void CudaAlgorithm::freeMemory()
{
	// If memory is allocated, free it
	if(allocated)
	{
		// freeCudaVector(floatBuffer);
		// freeCudaVector(windowBuffer);
		// freeCudaVector(complexBuffer);
		// freeCudaVector(charBuffer);
        allocated = false;
        printf("Free device memory\n");
	}
}


bool CudaAlgorithm::initDeviceEnv()
{
	//Allocate device memory for processing chain
	total_required_mem = (x_size * y_size * z_size * sizeof(float)*2
        + x_size * y_size * z_size * sizeof(cufftComplex)*2
        + x_size * y_size * z_size * color_depth * sizeof(unsigned char)
        + x_size * sizeof(float));

    printf("\nNeeded memory: %.2lf; free memory (%ld/%ld) MBytes\n",
        total_required_mem/(1024*1024),
        device->getFreeMemory()/(1024*1024),
        device->totalMemory()/(1024*1024));

	if(device->checkMemory(total_required_mem))
	{
		floatBuffer = new CudaVector<float>(device, x_size * y_size * z_size, true);
		complexBuffer = new CudaVector<cufftComplex>(device, x_size * y_size * z_size, true);
		charBuffer = new CudaVector<unsigned char>(device, x_size * y_size * z_size * color_depth, true);
		windowBuffer = new CudaVector<float>(device, x_size, true);
		allocated = true;
		return 1;
	}
	else
	{
        floatBuffer = NULL;
        complexBuffer = NULL;
        charBuffer = NULL;
        windowBuffer = NULL;
		printf("Not enough memory avaible... \n");
		return 0;
	}

}

void CudaAlgorithm::rangeMap(float* idata, char* odata, winType type, numKind kind, color_t colormap)
{
    CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(), idata, x_size*y_size*sizeof(float), cudaMemcpyHostToDevice));
    complexBuffer->resize((x_size/2+1) * y_size);
    charBuffer->resize((x_size/2+1) * y_size * color_depth);
    if(kind == COMPLEX)
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size/2+1, type);
        base->hilbertTransform(floatBuffer->getDevPtr(), complexBuffer->getDevPtr(), x_size, y_size);
        base->window(complexBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size/2+1, y_size);
        base->transpose(complexBuffer->getDevPtr(), x_size/2+1, y_size);
        base->window(complexBuffer->getDevPtr(), windowBuffer->getDevPtr(), y_size, x_size/2+1);
        base->transpose(complexBuffer->getDevPtr(), y_size, x_size/2+1);
        base->c2c1dFFT(complexBuffer->getDevPtr(), x_size/2+1, y_size);
    }
    else
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, type);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size);
        base->transpose(floatBuffer->getDevPtr(), x_size, y_size);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), y_size,x_size);
        base->transpose(floatBuffer->getDevPtr(), y_size, x_size);
        base->r2c1dFFT(complexBuffer->getDevPtr(), x_size, y_size, floatBuffer->getDevPtr());
    }
    floatBuffer->resize((x_size/2+1)*y_size);
    base->absolute(complexBuffer->getDevPtr(), floatBuffer->getDevPtr(), x_size/2+1, y_size);
    base->mapColors(floatBuffer->getDevPtr(), charBuffer->getDevPtr(), x_size/2+1, y_size, colormap);
    CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), charBuffer->getSize(), cudaMemcpyDeviceToHost));
}

void CudaAlgorithm::realtimeRangeMap(float* idata, char* odata, int nof_incoming_records, winType type, color_t colormap)
{
    // static variables for pipelining
    static int channel = 0;                                         // counts the current channel
    static int nof_processed_records = 0;                           // counter is inremented when recieved records are processed
    static long int* position = NULL;                               // position points to current position of data buffer

    int nfft = x_size/2+1;                                          // samples after R2C FFT or hilbertransform
    int offset = nof_processed_records * (nfft-2);                  // offset to handle data before FFT, becuse the data size is larger before the FFT

    // For each channel/z_size it is required to have position pointer
    if(position == NULL)
        position = (long int*)malloc(z_size*sizeof(long int));

    // calculate the current position (first element) of data to be processed in the current run
    position[channel] = channel * y_size * nfft + nof_processed_records * nfft;

    // Copy incoming data to GPU
    CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(position[channel] + offset), idata,
        x_size*nof_incoming_records*sizeof(float), cudaMemcpyHostToDevice));

    // Start processing of incoming data
    base->setWindow(windowBuffer->getDevPtr(), x_size, type);
    base->window(floatBuffer->getDevPtr(position[channel] + offset), windowBuffer->getDevPtr(), x_size, nof_incoming_records);
    base->r2c1dFFT(complexBuffer->getDevPtr(position[channel]), x_size, nof_incoming_records, floatBuffer->getDevPtr(position[channel] + offset));
    base->absolute(complexBuffer->getDevPtr(position[channel]), floatBuffer->getDevPtr(position[channel]), nfft, y_size);
    // If enoguh records are collected for the slow time FFT
    if(y_size-nof_incoming_records == nof_processed_records)
    {
        position[channel] = channel * nfft * y_size;
        base->mapColors(floatBuffer->getDevPtr(position[channel]), charBuffer->getDevPtr(position[channel]),nfft, y_size, colormap);
        CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), nfft * y_size * color_depth, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        // Reset variables because images are processed
        if(channel == z_size-1)
        {
            channel = 0;
            nof_processed_records = 0;
        }
        else
        {
            channel++;
        }
        return;
    }
    if(channel == z_size-1)
    {
        channel = 0;
        nof_processed_records += nof_incoming_records;
    }
    else
        channel++;
}

void CudaAlgorithm::rangeDopplerMap(float* idata, char* odata, winType type, numKind kind, color_t colormap)
{
    CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(), idata, x_size*y_size*sizeof(float), cudaMemcpyHostToDevice));
    complexBuffer->resize((x_size/2+1) * y_size);
    charBuffer->resize((x_size/2+1) * y_size * color_depth);
    if(kind == COMPLEX)
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size/2+1, type);
        base->hilbertTransform(floatBuffer->getDevPtr(), complexBuffer->getDevPtr(), x_size, y_size);
        base->window(complexBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size/2+1, y_size);
        base->transpose(complexBuffer->getDevPtr(), x_size/2+1, y_size);
        base->window(complexBuffer->getDevPtr(), windowBuffer->getDevPtr(), y_size, x_size/2+1);
        base->transpose(complexBuffer->getDevPtr(), y_size, x_size/2+1);
        base->c2c1dFFT(complexBuffer->getDevPtr(), x_size/2+1, y_size);
    }
    else
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, type);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size);
        base->transpose(floatBuffer->getDevPtr(), x_size, y_size);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), y_size,x_size);
        base->transpose(floatBuffer->getDevPtr(), y_size, x_size);
        base->r2c1dFFT(complexBuffer->getDevPtr(), x_size, y_size, floatBuffer->getDevPtr());
    }
    floatBuffer->resize((x_size/2+1)*y_size);
    base->hermitianTranspose(complexBuffer->getDevPtr(), x_size/2+1, y_size);
    base->c2c1dFFT(complexBuffer->getDevPtr(), y_size, x_size/2+1);
    base->hermitianTranspose(complexBuffer->getDevPtr(), y_size, x_size/2+1);
    base->absolute(complexBuffer->getDevPtr(), floatBuffer->getDevPtr(), x_size/2+1, y_size);
    base->mapColors(floatBuffer->getDevPtr(), charBuffer->getDevPtr(), x_size/2+1, y_size, colormap);
    CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), charBuffer->getSize(), cudaMemcpyDeviceToHost));
}



// In realtime applications it is may required to pipeline the incoming data.
void CudaAlgorithm::realtimeRangeDopplerMap(float* idata, char* odata, int nof_incoming_records, winType type, color_t colormap)
{
    // static variables for pipelining
    static int channel = 0;                                         // counts the current channel
    static int nof_processed_records = 0;                           // counter is inremented when recieved records are processed
    static long int* position = NULL;                               // position points to current position of data buffer

    int nfft = x_size/2+1;                                          // samples after R2C FFT or hilbertransform
    int offset = nof_processed_records * (nfft-2);                  // offset to handle data before FFT, becuse the data size is larger before the FFT

    // For each channel/z_size it is required to have position pointer
    if(position == NULL)
        position = (long int*)malloc(z_size*sizeof(long int));

    // calculate the current position (first element) of data to be processed in the current run
    position[channel] = channel * y_size * nfft + nof_processed_records * nfft;

    // Copy incoming data to GPU
    CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(position[channel] + offset), idata,
        x_size*nof_incoming_records*sizeof(float), cudaMemcpyHostToDevice));

    // Start processing of incoming data
    base->setWindow(windowBuffer->getDevPtr(), x_size, type);
    base->window(floatBuffer->getDevPtr(position[channel] + offset), windowBuffer->getDevPtr(), x_size, nof_incoming_records);
    base->r2c1dFFT(complexBuffer->getDevPtr(position[channel]), x_size, nof_incoming_records, floatBuffer->getDevPtr(position[channel] + offset));
    // If enoguh records are collected for the slow time FFT
    // Before perfoming the second FFT it is necessary to collect all needed data.
    if(y_size-nof_incoming_records == nof_processed_records)
    {
        position[channel] = channel * nfft * y_size;
        base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), nfft, y_size);
        base->c2c1dFFT(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
        base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
        base->absolute(complexBuffer->getDevPtr(position[channel]), floatBuffer->getDevPtr(position[channel]), nfft, y_size);
        base->mapColors(floatBuffer->getDevPtr(position[channel]), charBuffer->getDevPtr(position[channel]),nfft, y_size, colormap);
        CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), nfft * y_size * color_depth, cudaMemcpyDeviceToHost));
        // Reset variables because images are processed
        if(channel == z_size-1)
        {
            channel = 0;
            nof_processed_records = 0;
        }
        else
        {
            channel++;
        }
        return;
    }
    if(channel == z_size-1)
    {
        channel = 0;
        nof_processed_records += nof_incoming_records;
    }
    else
        channel++;
}


// In realtime applications it is may required to pipeline the incoming data.
// void CudaAlgorithm::streamingRangeDopplerMap(float* idata, char* odata, int nof_incoming_records, winType type, color_t colormap)
// {
//     int nfft = x_size/2+1;                                          // samples after R2C FFT or hilbertransform
//     int offset = nof_processed_records * (nfft-2);                  // offset to handle data before FFT, becuse the data size is larger before the FFT
//
//     cudaStream_t* stream = (cudaStream_t*)malloc(z_size*sizeof(cudaStream_t));
//
//     // For each channel/z_size it is required to have position pointer
//     if(position == NULL)
//         position = (long int*)malloc(z_size*sizeof(long int));
//
//     // Copy incoming data to GPU
//     for(int ch = 0; ch < y_size; ch++)
//     {
//         // calculate the current position (first element) of data to be processed in the current run
//         position[ch] = ch * y_size * nfft + nof_processed_records * nfft;
//         CUDA_CHECK(cudaMemcpyAsync(floatBuffer->getDevPtr(position[ch] + offset), idata,
//             x_size*nof_incoming_records*sizeof(float), cudaMemcpyHostToDevice, stream[ch]));
//     }
//
//     // Start processing of incoming data
//     base->setWindow(windowBuffer->getDevPtr(), x_size, type);
//     base->window(floatBuffer->getDevPtr(position[channel] + offset), windowBuffer->getDevPtr(), x_size, nof_incoming_records);
//     base->r2c1dFFT(complexBuffer->getDevPtr(position[channel]), x_size, nof_incoming_records, floatBuffer->getDevPtr(position[channel] + offset));
//     // If enoguh records are collected for the slow time FFT
//     // Before perfoming the second FFT it is necessary to collect all needed data.
//     if(y_size-nof_incoming_records == nof_processed_records)
//     {
//         position[channel] = channel * nfft * y_size;
//         base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), nfft, y_size);
//         base->c2c1dFFT(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
//         base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
//         base->absolute(complexBuffer->getDevPtr(position[channel]), floatBuffer->getDevPtr(position[channel]), nfft, y_size);
//         base->mapColors(floatBuffer->getDevPtr(position[channel]), charBuffer->getDevPtr(position[channel]),nfft, y_size, colormap);
//         // CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), nfft * y_size * color_depth, cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]+nfft * y_size * color_depth/2), nfft * y_size * color_depth/2, cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(&odata[nfft * y_size * color_depth/2], charBuffer->getDevPtr(position[channel]), nfft * y_size * color_depth/2, cudaMemcpyDeviceToHost));
//         // Reset variables because images are processed
//         if(channel == z_size-1)
//         {
//             channel = 0;
//             nof_processed_records = 0;
//         }
//         else
//         {
//             channel++;
//         }
//         return;
//     }
//     if(channel == z_size-1)
//     {
//         channel = 0;
//         nof_processed_records += nof_incoming_records;
//     }
//     else
//         channel++;
// }
