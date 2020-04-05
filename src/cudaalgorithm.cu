#include "cudaalgorithm.cuh"

CudaAlgorithm::CudaAlgorithm(CudaGPU* dev, processing_conf *prc)
{
    device = dev;
    conf = prc;
    base = new CudaBase(device);
    if(prc)
	{
        x_size = conf->x;
        y_size = conf->y;
    	z_size = conf->z;
    }
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
		floatBuffer->freeMemory();
		windowBuffer->freeMemory();
		complexBuffer->freeMemory();
		charBuffer->freeMemory();
        shortBuffer->freeMemory();
        if(conf->image_format == JPG)
            base->destroyJpegEncoder();
        allocated = false;
        printf("Free device memory\n");
	}
}


bool CudaAlgorithm::initProcessingEnv()
{
	//Allocate device memory for processing chain
    total_required_mem = (x_size * y_size * z_size * sizeof(float)*2
            + x_size * y_size * z_size * sizeof(cufftComplex)*2
            + x_size * y_size * z_size * sizeof(unsigned char)
            + x_size * sizeof(float));
    printf("\nNeeded memory: %.2lf; free memory (%ld/%ld) MBytes\n",
        total_required_mem/(1024*1024),
        device->getFreeMemory()/(1024*1024),
        device->totalMemory()/(1024*1024));

	//if(device->checkMemory(total_required_mem))
	//{
		floatBuffer = new CudaVector<float>(device, x_size * y_size * z_size, true);
        shortBuffer = new CudaVector<short>(device, x_size * y_size * z_size, true);
		complexBuffer = new CudaVector<cufftComplex>(device, x_size * y_size * z_size, true);
		charBuffer = new CudaVector<unsigned char>(device, x_size * y_size * z_size * 3, true);
		windowBuffer = new CudaVector<float>(device, x_size*y_size, true);
        if(conf->image_format == JPG)
            base->initJpegEncoder(x_size/2+1, y_size);
		allocated = true;
		return 1;
	//}
	//else
	//{
        //floatBuffer = NULL;
        //complexBuffer = NULL;
        //charBuffer = NULL;
        //windowBuffer = NULL;
		//printf("Not enough memory avaible... \n");
		//return 0;
	//}

}

// void CudaAlgorithm::make_tcp_header(tcp_header* header, int records_processed, int nof_records, int ch)
// {
//
//     header->total_size = htonl(image_size);
//     header->total_nof_records = htonl(nof_records);
//     header->rec_records = htonl(records_processed);
//     header->nof_channels = htonl(z_size);
//     header->current_channel = htonl(ch);
//     header->img_height = htonl(y_size);
//     header->img_width = htonl(x_size/2+1);
//     header->format = htonl(JPG);
// }


template <typename T>
void CudaAlgorithm::process(T* idata, char* odata)
{
    switch(conf->alg)
    {
        case RANGE:
            rangeMap(idata, odata);
            break;
        case RANGE_DOPPLER:
            rangeDopplerMap(idata, odata);
            break;
    }
}
template void CudaAlgorithm::process<float>(float*, char*);
template void CudaAlgorithm::process<short>(short*, char*);

template <typename T>
void CudaAlgorithm::rangeMap(T* idata, char* odata)
{
    int nfft = x_size/2+1;
    //cout << "Processing range map" << endl;
    if(std::is_same<T, short>::value)
    {
        CUDA_CHECK(cudaMemcpy(shortBuffer->getDevPtr(), idata,x_size*y_size*sizeof(T), cudaMemcpyHostToDevice));
        base->convert(shortBuffer->getDevPtr(), floatBuffer->getDevPtr(),x_size*y_size);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(), idata,x_size*y_size*sizeof(T), cudaMemcpyHostToDevice));
    }
    // If desired execute 2d window
    if(conf->window_2d)
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window, y_size);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size, 2);
    }
    // Otherwise 1d window
    else
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size, 1);
    }
    base->r2c1dFFT(complexBuffer->getDevPtr(), x_size, y_size, floatBuffer->getDevPtr());
    base->absolute(complexBuffer->getDevPtr(), floatBuffer->getDevPtr(), nfft, y_size);
    base->mapColors(floatBuffer->getDevPtr(), charBuffer->getDevPtr(), nfft, y_size, conf->color_mapping, conf->log_mapping);
    if(conf->image_format == JPG)
    {
        base->encodeBmpToJpeg((unsigned char*)charBuffer->getDevPtr(), (uint8_t*)odata, &image_size, save_path);
        CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), image_size, cudaMemcpyDeviceToHost));
    }
    else if(conf->image_format == BMP24)
    {
        CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), nfft*y_size*3, cudaMemcpyDeviceToHost));
        Bitmap_IO img(nfft, y_size, 24, false);
        img.setImage(odata);
        img.save(save_path+".bmp");
    }
    //cout << "Copied data GPU -> Host" << endl;
}
template void CudaAlgorithm::rangeMap<float>(float*, char*);
template void CudaAlgorithm::rangeMap<short>(short*, char*);


void CudaAlgorithm::realtimeRangeMap(float* idata, char* odata, int nof_incoming_records, int type, int colormap)
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
    base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window);
    base->window(floatBuffer->getDevPtr(position[channel] + offset), windowBuffer->getDevPtr(), x_size, nof_incoming_records);
    base->r2c1dFFT(complexBuffer->getDevPtr(position[channel]), x_size, nof_incoming_records, floatBuffer->getDevPtr(position[channel] + offset));
    base->absolute(complexBuffer->getDevPtr(position[channel]), floatBuffer->getDevPtr(position[channel]), nfft, y_size);
    // If enoguh records are collected for the slow time FFT
    if(y_size-nof_incoming_records == nof_processed_records)
    {
        position[channel] = channel * nfft * y_size;
        base->mapColors(floatBuffer->getDevPtr(position[channel]), charBuffer->getDevPtr(position[channel]),nfft, y_size, conf->color_mapping, conf->log_mapping);
        if(conf->image_format == JPG)
        {
            base->encodeBmpToJpeg((unsigned char*)charBuffer->getDevPtr(position[channel]), (uint8_t*)odata, &image_size);
            CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), image_size, cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), nfft * y_size * 3, cudaMemcpyDeviceToHost));
        }
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
template <typename T>
void CudaAlgorithm::rangeDopplerMap(T* idata, char* odata)
{
    //cout << "Processing range map" << endl;
    int nfft = x_size/2+1;
    //cout << "Processing range doppler map" << endl;

    if(std::is_same<T, short>::value)
    {

        CUDA_CHECK(cudaMemcpy(shortBuffer->getDevPtr(), idata,x_size*y_size*sizeof(T), cudaMemcpyHostToDevice));
        base->convert(shortBuffer->getDevPtr(), floatBuffer->getDevPtr(),x_size*y_size);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(), idata,x_size*y_size*sizeof(T), cudaMemcpyHostToDevice));
    }
    // If desired execute 2d window
    if(conf->window_2d)
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window, y_size);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size, 2);
    }
    // Otherwise 1d window
    else
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window);
        base->window(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size, 1);
    }
    base->r2c1dFFT(complexBuffer->getDevPtr(), x_size, y_size, floatBuffer->getDevPtr());
    //floatBuffer->resize((x_size/2+1)*y_size);CUDA_CHECK(cudaDeviceSynchronize());
    base->hermitianTranspose(complexBuffer->getDevPtr(),nfft, y_size);
    base->c2c1dFFT(complexBuffer->getDevPtr(), y_size, nfft);
    base->fftshift(complexBuffer->getDevPtr(), y_size, nfft);
    base->hermitianTranspose(complexBuffer->getDevPtr(), y_size, nfft);
    if(conf->dop_zero_sup)
        base->zeroFilling(complexBuffer->getDevPtr(), y_size/2, nfft,conf->dop_zero_range);
    base->absolute(complexBuffer->getDevPtr(), floatBuffer->getDevPtr(), nfft, y_size);
    base->mapColors(floatBuffer->getDevPtr(), charBuffer->getDevPtr(), x_size, y_size, conf->color_mapping, conf->log_mapping);
    if(conf->image_format == JPG)
    {
        base->encodeBmpToJpeg((unsigned char*)charBuffer->getDevPtr(), (uint8_t*)odata, &image_size, save_path);
        CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), image_size, cudaMemcpyDeviceToHost));
    }
    else if(conf->image_format == BMP24)
    {
        CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), nfft*y_size*3, cudaMemcpyDeviceToHost));
        Bitmap_IO img(nfft, y_size, 24, false);
        img.setImage(odata);
        img.save(save_path+".bmp");
    }
}
template void CudaAlgorithm::rangeDopplerMap<float>(float*, char*);
template void CudaAlgorithm::rangeDopplerMap<short>(short*, char*);


// In realtime applications it is may required to pipeline the incoming data.
void CudaAlgorithm::realtimeRangeDopplerMap(float* idata, char* odata, int nof_incoming_records, int type, int colormap)
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
    base->setWindow(windowBuffer->getDevPtr(), x_size, conf->window);
    base->window(floatBuffer->getDevPtr(position[channel] + offset), windowBuffer->getDevPtr(), x_size, nof_incoming_records);
    base->r2c1dFFT(complexBuffer->getDevPtr(position[channel]), x_size, nof_incoming_records, floatBuffer->getDevPtr(position[channel] + offset));
    // If enoguh records are collected for the slow time FFT
    // Before perfoming the second FFT it is necessary to collect all needed data.
    if(y_size-nof_incoming_records == nof_processed_records)
    {
        position[channel] = channel * nfft * y_size;
        base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), nfft, y_size);
        base->c2c1dFFT(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
        base->fftshift(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
        base->hermitianTranspose(complexBuffer->getDevPtr(position[channel]), y_size, nfft);
        base->absolute(complexBuffer->getDevPtr(position[channel]), floatBuffer->getDevPtr(position[channel]), nfft, y_size);
        base->mapColors(floatBuffer->getDevPtr(position[channel]), charBuffer->getDevPtr(position[channel]),nfft, y_size, conf->color_mapping, conf->log_mapping);
        if(conf->image_format == JPG)
        {
            base->encodeBmpToJpeg((unsigned char*)charBuffer->getDevPtr(position[channel]), (uint8_t*)odata, &image_size);
            CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), image_size, cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CHECK(cudaMemcpy(&odata[0], charBuffer->getDevPtr(position[channel]), nfft * y_size * 3, cudaMemcpyDeviceToHost));
        }
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

template<typename T> void CudaAlgorithm::save(CudaVector<T>* vec, string filename)
{
    FILE* fid;
    T* h_buffer = (T*)malloc(vec->getSize());
    int width;
    unsigned int _size = (float)sizeof(T);

    CUDA_CHECK(cudaMemcpy(h_buffer, vec->getDevPtr(), vec->getSize(), cudaMemcpyDeviceToHost));

    fid = fopen(filename.c_str(), "wb");
    fwrite((void*)&z_size, sizeof(z_size), 1, fid);
    fwrite((void*)&y_size, sizeof(y_size), 1, fid);
    fwrite((void*)&x_size, sizeof(width), 1, fid);
    fwrite((void*)&_size, sizeof(_size), 1, fid);
    fwrite((void*)h_buffer, _size, vec->getSize()/_size, fid);
    fclose(fid);
}
template void CudaAlgorithm::save<cufftComplex>(CudaVector<cufftComplex>* vec, string filename);
template void CudaAlgorithm::save<float>(CudaVector<float>* vec, string filename);
template void CudaAlgorithm::save<int>(CudaVector<int>* vec, string filename);
template void CudaAlgorithm::save<short>(CudaVector<short>* vec, string filename);
template void CudaAlgorithm::save<char>(CudaVector<char>* vec, string filename);
template void CudaAlgorithm::save<double>(CudaVector<double>* vec, string filename);


void CudaAlgorithm::insertProcessingConf()
{
    conf = new processing_conf;
    msg("\nIt is required to specify the processing configuration. \nPlease insert the configuration arguments!");
    msg("Choose an algorithm:\n  1:Range\n  2:Range-Doppler");
    cin >> conf->alg;
    msg("Choose a window function:\n  1:Hamming\n  2:Hann\n  3:Bartlett\n  4:Blackman");
    cin >> conf->window;
    msg("Choose a dimension for the window function?\n 0:one-dimensional\n 1:two-dimensional");
    cin >> conf->window_2d;
    if(conf->alg == RANGE_DOPPLER)
    {
        msg("Do you want to suppress stationary objects?\n 0: no\n 1: yes");
        cin >> conf->dop_zero_sup;
        if(conf->dop_zero_sup)
        {
            msg("Insert a suppresion range: ");
            cin >> conf->dop_zero_range;
        }
    }
    msg("Choose a colormapping:\n  1:Jet\n  2:Viridis\n  3:Accent\n  4:Magma\n  5:Inferno\n  6:Blue");
    cin >> conf->color_mapping;
    msg("Do you want a logarithmic scale?\n  0:no\n  1:yes");
    cin >> conf->log_mapping;
    msg("What kind of image should generated?\n  0:None \n  1:BMP24\n  2:JPG");
    cin >> conf->image_format;
    msg("Insert the image height in pixel: ");
    cin >> conf->y;
    y_size = conf->y;
}

void CudaAlgorithm::printConfiguration()
{
    msg("Processing configuration:");
    msg("Algorithm: " + to_string(conf->alg));
    msg("Doppler zero suppresion: " + to_string(conf->dop_zero_sup));
    msg("Doppler zero range: " + to_string(conf->dop_zero_range));
    msg("Window function:" + to_string(conf->window));
    msg("2d or 1d:" + to_string(conf->window_2d));
    msg("color mapping:" + to_string(conf->color_mapping));
    msg("image format:" + to_string(conf->image_format));
    msg("log mapping:" + to_string(conf->log_mapping));
    msg("Width:" + to_string(x_size));
    msg("Height:" + to_string(y_size));
    msg("Depth:" + to_string(z_size));
}
