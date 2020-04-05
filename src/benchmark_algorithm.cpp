#include "cudaalgorithm.cuh"
#include "utils.h"

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <cuda_profiler_api.h>

using namespace std;
using namespace utils;

int main(int argc, char** argv)
{
    int height;
    int width;
    int runs;

    double timing;
    double avg_time;
    double max_time = 0;;
    double min_time = 99999;;

    clock_t start, end;

    short *host_idata;
    char* host_odata;
    CudaGPU device(0);
	CudaAlgorithm cu_algo(&device);

    cu_algo.insertProcessingConf();
    msg("Insert samples in record: ");
    cin >> width;
    msg("Insert number of runs (the more, the accurate is the result): ");
    cin >> runs;
    height = cu_algo.getHeight();
    cu_algo.setSavePath("");        // prevent algorithm to store results
    cu_algo.setWidth(width);
	cu_algo.setDepth(1);

	msg("Starting benchmark of algorithm...");

    host_idata = (short*)malloc(width*height*sizeof(short));
    host_odata = (char*)malloc(width*height*sizeof(char));

    if(cu_algo.initProcessingEnv())
	{
        for(int i = 0; i < runs; i++)
        {
            start = clock();
            cu_algo.process(host_idata, host_odata);
            cudaDeviceSynchronize();
			end = clock();
			timing += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;
            if(max_time < ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC)
                max_time = ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;
            if(min_time > ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC)
                min_time = ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;
        }
        avg_time = timing / runs;
        msg("Average time needed for range-Doppler algorithm: " + to_string(avg_time));
        msg("Max for range-Doppler algorithm: " + to_string(max_time));
        msg("Min for range-Doppler algorithm: " + to_string(min_time));
    }
    return 0;
}
