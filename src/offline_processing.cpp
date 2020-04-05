#include "signalgenerator.h"
#include "cudaalgorithm.cuh"
#include "bitmap_io.h"
#include "utils.h"

#include <stdio.h>      /* puts, printf */
#include <time.h>       /* time_t, struct tm, time, localtime */
#include <string>
#include <unistd.h>
#include <sys/types.h>

using namespace std;
using namespace utils;

int main(int argc, char** argv)
{
    std::string filename;
    int nof_samples;
    int img_height;
    int nof_records;
    clock_t start, end;

    SignalGenerator *generator = new SignalGenerator();
    msg("Welcome!\nThis program allows you to process range or range-Doppler maps. \nA binary file is required that has a defined structure. \nHow the structure is organized is described in the read.md file.\n\nPlease insert the entire directory including the filename.");
    cin >> filename;
    msg("Insert the number of records to read (per channel): ");
    cin >> nof_records;
    if(!(generator->load(filename, nof_records)))
    {
        msg("Program gets aborted");
        return 0;
    }
    // GPU setup
    CudaGPU *device = new CudaGPU(0);                  // Initialize the first (0) CUDA capable device
    CudaAlgorithm algthm(device);                       // Instantiate class for implemented algorithms
    algthm.insertProcessingConf();
    algthm.setWidth(generator->getLength());
    algthm.setDepth(generator->getChannels());
    algthm.initProcessingEnv();             // Allocate memory for algorithm and intialize further enviroment things
    img_height = algthm.getHeight();
    nof_samples = algthm.getWidth();
    Bitmap_IO *bmp = new Bitmap_IO(nof_samples/2+1, img_height, 24);

    cout << "Processing Signal" << endl;
    for( int i = 0; i < (int)nof_records/img_height; i++ )
    {
        for(int ch = 0; ch < generator->getChannels(); ch++)
        {
            cout << i <<" Map processed, on channel "<< ch << endl;
            start = clock();
            algthm.setSavePath("./results/img/CH"+to_string(ch)+"_num_"+to_string(i));
            algthm.process(&generator->getLoadedSignal(ch)[i*img_height*nof_samples],bmp->getImagePtr());
            cudaDeviceSynchronize();
            end = clock();
            cout << "Elapsed time " << to_string(((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC) << " ms"<< endl;
        }
    }
    return 0;
}
