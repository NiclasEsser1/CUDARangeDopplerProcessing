#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaAlgorithm.cuh"
#include "CudaTest.h"
#include "Bitmap_IO.h"

#include <stdio.h>      /* puts, printf */
#include <time.h>       /* time_t, struct tm, time, localtime */
#include <string>

int main()
{
    // Acquisition parameters (simulated)
    int nof_samples = 2048;             // samples per record
    int nof_records = 4096;             // total amount of records
    int nof_records_packet = 512;       // records to be transmitted per run
    int nof_channels = 4;               // amount of operating channels
    int nof_recieved_records = 0;       //
    int recieved_all_records = 0;       // flag for while loop
    int runs = 1;                       // counts runs per image

    // Visualization parameters
    int img_height = 1024;              // pixel
    int img_width = nof_samples/2+1;    // pixel

    // Signal parameters
    float fsample = 100000000;          // Hz
    float fcenter = 6250000;            // Hz
    float amplitude = 10;               // Volt
    float bw = 6000000;                 // bandwidth in Hz
    float time = 0.0001;                // seconds
    float fdoppler = 5000;              // Hz

    // Processing options
    winType window = HAMMING;           // window function type
    color_t colormap = JET;             // Colormap to choose

    // GPU setup
    CudaGPU device(0);                  // Initialize the first (0) CUDA capable device
    CudaBase cuda_base(&device);        // Instantiate base class for CUDA launcher
	CudaAlgorithm algthm(&cuda_base,    // Instantiate class for implemented algorithms
        nof_samples, img_height, nof_channels, 3);

    algthm.initDeviceEnv();             // Allocate memory for algorithm and intialize further enviroment things


    SignalGenerator signal(fsample,     // Generate signals on CPU as simulated signal
        fcenter, amplitude, nof_samples, nof_records_packet, nof_channels);

    // Initialize images for range doppler maps
    int image_num = 0;                  // counter for filename
    char image_dir[100];                // directory of images
    Bitmap_IO** images;                 // multidimensional class pointer for bitmap images


    // Allocate bitmap instances for each channel
    images = (Bitmap_IO**)malloc(nof_channels*sizeof(Bitmap_IO*));
    // Instantiate bitmap images
    for(int ch = 0; ch < nof_channels; ch++)
        images[ch] = new Bitmap_IO(img_width, img_height, algthm.getColorDepth()*8);

    // Verify if GPU is capable to handle cuda streams
    if(!device.getProperties().deviceOverlap)
    {
        printf("Device will not handle overlaps, so cuda streams are not supported\n");
        return 0;
    }

    // Loop until all records are recieved
    while(!recieved_all_records)
    {
        // For each channel acquire and process data, (sequentiell)
        for(int ch = 0; ch < nof_channels; ch++)
        {
            // acquire data
            signal.sweep(bw, time, fdoppler*ch, true, runs); // Generates a sweep with noise (true)
            // signal.save("signal"+std::to_string(ch)+".dat",bw,time);

            // process data to range doppler maps
            algthm.realtimeRangeMap(signal.getSignal(),
                images[ch]->GetImagePtr(), nof_records_packet,
                window, colormap);
            CUDA_CHECK(cudaDeviceSynchronize());
            // If maps are processed
            if(nof_recieved_records >= img_height && nof_recieved_records%img_height==0)
            {
                sprintf(image_dir, "./results/img/realtime/CH%d_%d_rangedoppler.bmp", ch, image_num);
                images[ch]->Save(image_dir);
                runs = 0;
                if(ch == nof_channels-1)
                    image_num++;
            }
        }
        runs++;
        if(nof_recieved_records == nof_records)
            recieved_all_records = 1;
        nof_recieved_records += nof_records_packet;
        if(nof_recieved_records%img_height == 0)
            printf("Records processed: %d/%d\n",nof_recieved_records, nof_records);
    }
    return 0;
}
