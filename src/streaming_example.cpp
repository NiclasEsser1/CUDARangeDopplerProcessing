#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaAlgorithm.cuh"
#include "CudaTest.h"
#include "Bitmap_IO.h"
#include "Socket.h"

#include <stdio.h>      /* puts, printf */
#include <time.h>       /* time_t, struct tm, time, localtime */
#include <string>
#include <cuda_profiler_api.h>

int main(int argc, char* argv[])
{
    // Acquisition parameters (simulated)
    int nof_samples = 2048;             // samples per record
    int nof_records = 8192;             // total amount of records
    int nof_records_packet = 512;       // records to be transmitted per run
    unsigned nof_channels = 3;               // amount of operating channels
    unsigned nof_recieved_records = 0;       //
    int recieved_all_records = 0;       // flag for while loop
    int runs = 1;                       // counts runs per image

    // Visualization parameters
    unsigned img_height = 512;              // pixel
    unsigned img_width = nof_samples/2+1;    // pixel
    unsigned img_color_depth = 3;
	char image_dir[256];
    // Variables for TCP Streaminig
    Socket socket;
    tcp_header header;

    // Signal parameters
    float fsample = 100000000;          // Hz
    float fcenter = 6250000;            // Hz
    float amplitude = 10;               // Volt
    float bw = 6000000;                 // bandwidth in Hz
    float time = 0.0001;                // seconds
    float fdoppler = 5000;              // Hz

    // Processing options
    int window = HAMMING;           // window function type
    color_t colormap = JET;             // Colormap to choose

    // GPU setup
    CudaGPU device(0);                  // Initialize the first (0) CUDA capable device
    CudaBase cuda_base(&device);        // Instantiate base class for CUDA launcher
	CudaAlgorithm algthm(&cuda_base,    // Instantiate class for implemented algorithms
        nof_samples, img_height, nof_channels, 3);

    algthm.initDeviceEnv();             // Allocate memory for algorithm and intialize further enviroment things

    // Simulated signals
    SignalGenerator **signals;


    // Initialize images for range doppler maps
    uint8_t** images;             // pointer-pointer to image buffers


    // Allocate bitmap instances for each channel
    images = (uint8_t**)malloc(nof_channels*sizeof(uint8_t*));
    signals = (SignalGenerator**)malloc(nof_channels*sizeof(SignalGenerator*));
    for(int ch = 0; ch < nof_channels; ch++)
    {
        images[ch] = (uint8_t*) malloc(img_width*img_height);
        // Generate signals on CPU as simulated signal
        signals[ch] = new SignalGenerator(fsample, fcenter, amplitude,
            nof_samples, nof_records_packet, nof_channels);
        // Generates a sweep with noise (true)
        // signals[ch]->sweep(bw, time, fdoppler*ch, true);
    }

    // Setup an tcp socket server for Inter Process Communication between, back and front end
    socket.open();
    cudaProfilerStart();
    // Loop until all records are recieved
    while(!recieved_all_records)
    {
        nof_recieved_records += nof_records_packet;
        // For each channel acquire and process data, (sequentiell)
        for(int ch = 0; ch < nof_channels; ch++)
        {
            // process data to range doppler maps
            signals[ch]->sweep(bw, time, fdoppler*ch, true, runs);
            algthm.realtimeRangeMap(signals[ch]->getSignal(),images[ch],
                nof_records_packet,
                window, colormap);
            CUDA_CHECK(cudaDeviceSynchronize());
            // If maps are processed
            if(nof_recieved_records % img_height == 0)
            {
                if(socket.isActive())
                {
                    algthm.make_tcp_header(&header, nof_recieved_records, nof_records, ch);
                    socket.send(&header, sizeof(header));
                    socket.wait();
                    socket.send(images[ch], algthm.getImageSize());
                    socket.wait();
                }
                runs = 0;
            }
        }
        runs++;
        if(nof_recieved_records >= nof_records)
            recieved_all_records = 1;

        if(nof_recieved_records%img_height == 0)
            printf("Records processed: %d/%d\n",nof_recieved_records, nof_records);
    }
    cudaProfilerStop();
    cudaDeviceReset();
    return 0;
}
