#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaAlgorithm.cuh"
#include "CudaTest.h"
#include "Bitmap_IO.h"
#include "Socket.h"

#include <stdio.h>      /* puts, printf */
#include <time.h>       /* time_t, struct tm, time, localtime */
#include <string>

int main(int argc, char* argv[])
{
    // Acquisition parameters (simulated)
    int nof_samples = 1024;             // samples per record
    int nof_records = 100000;             // total amount of records
    int nof_records_packet = 256;       // records to be transmitted per run
    unsigned nof_channels = 4;               // amount of operating channels
    unsigned nof_recieved_records = 0;       //
    int recieved_all_records = 0;       // flag for while loop
    int runs = 1;                       // counts runs per image

    // Visualization parameters
    unsigned img_height = 512;              // pixel
    unsigned img_width = nof_samples/2+1;    // pixel
    unsigned img_color_depth = 3;
    unsigned img_size = img_height * img_width * img_color_depth;

    // Variables for TCP Streaminig
    Socket socket;
    tcp_header header;
    header.total_size = htonl(img_size);
    header.rec_records = htonl(nof_records);
    header.nof_channels = htonl(nof_channels);
    header.img_height = htonl(img_height);
    header.img_width = htonl(img_width);
    header.color_depth = htonl(img_color_depth);

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

    // Simulated signals
    SignalGenerator **signals;


    // Initialize images for range doppler maps
    int image_num = 0;                  // counter for filename
    char image_dir[256];                // directory of images
    Bitmap_IO** images;                 // multidimensional class pointer for bitmap images


    // Allocate bitmap instances for each channel
    images = (Bitmap_IO**)malloc(nof_channels*sizeof(Bitmap_IO*));
    signals = (SignalGenerator**)malloc(nof_channels*sizeof(SignalGenerator*));
    for(int ch = 0; ch < nof_channels; ch++)
    {
        // Instantiate bitmap images
        images[ch] = new Bitmap_IO(img_width, img_height, algthm.getColorDepth()*8);
        // Generate signals on CPU as simulated signal
        signals[ch] = new SignalGenerator(fsample, fcenter, amplitude,
            nof_samples, nof_records_packet, nof_channels);
        // Generates a sweep with noise (true)
        // signals[ch]->sweep(bw, time, fdoppler*ch, true);
    }

    // Setup an tcp socket server for Inter Process Communication between, back and front end
    if(socket.open())
        socket.writeToServer(&header, 1);
    // Loop until all records are recieved
    while(!recieved_all_records)
    {
        // For each channel acquire and process data, (sequentiell)
        for(int ch = 0; ch < nof_channels; ch++)
        {
            // process data to range doppler maps
            signals[ch]->sweep(bw, time, fdoppler*ch, true, runs);
            algthm.realtimeRangeDopplerMap(signals[ch]->getSignal(),
                images[ch]->GetImagePtr(), nof_records_packet,
                window, colormap);
            CUDA_CHECK(cudaDeviceSynchronize());
            // If maps are processed
            if(nof_recieved_records >= img_height && nof_recieved_records%img_height==0)
            {
                if(socket.isActive())
                    socket.writeToServer(images[ch]->GetImagePtr(), img_size);
                sprintf(image_dir, "/home/niclas/SoftwareProjekte/Cuda/PerformanceComparsion/results/img/streaming/CH%d_%d_rangedoppler.bmp", ch, image_num);
                images[ch]->Save(image_dir);
                runs = 0;
                if(ch == nof_channels-1)
                {
                    image_num++;
                }
            }
        }
        runs++;
        if(nof_recieved_records >= nof_records)
            recieved_all_records = 1;
        nof_recieved_records += nof_records_packet;


        if(nof_recieved_records%img_height == 0)
            printf("Records processed: %d/%d\n",nof_recieved_records, nof_records);
    }

    return 0;
}
