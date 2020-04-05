#include "cudabase.cuh"
#include "utils.h"

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#define ITERATIONS ( 100 )
#define NOF_PROCESSING_STEPS 3
#define NOF_IMAGES 32

using namespace utils;
using namespace std;

int main(int argc, char** argv)
{
    int height[NOF_IMAGES];
	int width[NOF_IMAGES];
    int size[NOF_IMAGES];
	for(int i = 0; i < NOF_IMAGES; i++)
	{
		height[i] = 256*(i+1);
        width[i] = height[i];
		size[i] = width[i]*height[i];
	}

	float timing[NOF_PROCESSING_STEPS];
	float avg_time[NOF_PROCESSING_STEPS];
	clock_t start, end;

    CudaGPU device;
    CudaBase cu_base(&device);
    CudaVector<cufftReal>* real;
    CudaVector<cufftComplex>* cplx;

    FILE* fid;
    fid = fopen("./results/benchmarks/benchmark_fft.csv", "w");
    fprintf(fid, "Size,elements,FFT(R2C),FFT(C2C),IFFT(C2C)\n");
    msg("Starting simulation of processing steps... ");
    for(int k = 0; k < NOF_IMAGES; k++)
	{
		msg("Operating on a "+to_string(height[k])+" x "+to_string(width[k])+" image...");
        msg("Allocating...");
        real = new CudaVector<cufftReal>(&device, size[k], true);
        cplx = new CudaVector<cufftComplex>(&device, size[k], true);

		fprintf(fid, "%dx%d,%d,",height[k], width[k],size[k]);
		for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
		{
			avg_time[j] = 0;
			timing[j] = 0;
		}

		for(int i = 0; i < ITERATIONS; i++)
		{
			if(!(i%1))
				msg("\n\n________________\nRun ("+to_string(i)+"/"+to_string(ITERATIONS)+")");
            cu_base.random((float*)real->getDevPtr(), size[k], 3,3);

            start = clock();
			cu_base.r2c1dFFT(cplx->getDevPtr(), width[k], height[k], real->getDevPtr());
			CUDA_CHECK(cudaDeviceSynchronize());
			end = clock();
			timing[0] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cu_base.c2c1dFFT(cplx->getDevPtr(), width[k], height[k]);
			CUDA_CHECK(cudaDeviceSynchronize());
			end = clock();
			timing[1] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cu_base.c2c1dInverseFFT(cplx->getDevPtr(), width[k], height[k]);
			CUDA_CHECK(cudaDeviceSynchronize());
			end = clock();
			timing[2] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

		}
		for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
		{
			avg_time[j] = timing[j] / ITERATIONS;
			if(j == NOF_PROCESSING_STEPS -1)
				fprintf(fid, "%.4f \n", avg_time[j]);
			else
				fprintf(fid, "%.4f,", avg_time[j]);
		}
		if(k != ITERATIONS-1)
	    {
            delete(real);
            delete(cplx);
        }
		sleep(1);
	}
	fclose(fid);
    return 0;
}
