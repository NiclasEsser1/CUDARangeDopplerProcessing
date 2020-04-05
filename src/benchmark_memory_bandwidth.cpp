#include "cudaalgorithm.cuh"

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define ITERATIONS ( 10 )
#define NOF_PROCESSING_STEPS 5
#define NOF_IMAGES 20

int main(int argc, char** argv)
{
	int height[NOF_IMAGES];
	int width[NOF_IMAGES];
    int size[NOF_IMAGES];
	for(int k = 0; k <NOF_IMAGES; k++)
	{
		height[k] = 256*(k+1);
		width[k] = height[k];
        size[k]= height[k]*width[k]*sizeof(float);
	}
	int jpeg_size;
    float *h_mem;
	double timing[NOF_PROCESSING_STEPS];
	double avg_time[NOF_PROCESSING_STEPS];
	clock_t start, end;

	CudaGPU device(0);
    CudaVector<float> *d_mem;

	FILE* fid;
	fid = fopen("./results/benchmarks/benchmark_memory_bandwidth.csv", "w");
	fprintf(fid, "size,bytesize,allocation,free,host2device,device2host,device2device\n");
	printf("Starting simulation of memory benchmark... \n");

	for(int k = 0; k < NOF_IMAGES; k++)
	{
        fprintf(fid, "%dx%d,%d,",height[k], width[k], size[k]);

        h_mem = new float[size[k]];
        CudaVector<float> *d_mem2 = new CudaVector<float>(&device, size[k]);
        for(int i = 0; i < ITERATIONS; i++)
        {
            if(!(i%1))
                printf("\n\n________________\nRun (%d/%d) \n",i,ITERATIONS);


			start = clock();
			d_mem = new CudaVector<float>(&device, size[k]);
			cudaDeviceSynchronize();
			end = clock();
			timing[0] += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

            start = clock();
			d_mem->freeMemory();
			cudaDeviceSynchronize();
			end = clock();
			timing[1] += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

            d_mem = new CudaVector<float>(&device, size[k]);

            start = clock();
			CUDA_CHECK(cudaMemcpy(d_mem->getDevPtr(), h_mem, size[k], cudaMemcpyHostToDevice))
			cudaDeviceSynchronize();
			end = clock();
			timing[2] += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

            start = clock();
			CUDA_CHECK(cudaMemcpy(h_mem, d_mem->getDevPtr(), size[k], cudaMemcpyDeviceToHost))
			cudaDeviceSynchronize();
			end = clock();
			timing[3] += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

            start = clock();
			CUDA_CHECK(cudaMemcpy(d_mem2->getDevPtr(), d_mem->getDevPtr(), size[k], cudaMemcpyDeviceToDevice))
			cudaDeviceSynchronize();
			end = clock();
			timing[4] += ((double)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;


            d_mem->freeMemory();
        }
        d_mem2->freeMemory();
        free(h_mem);
        for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
        {
            avg_time[j] = timing[j] / ITERATIONS;
            if(j == NOF_PROCESSING_STEPS -1)
                fprintf(fid, "%.4f \n", avg_time[j]);
            else
                fprintf(fid, "%.4f,", avg_time[j]);
        }
        sleep(0.5);
    }
    return 0;
}
