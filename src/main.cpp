#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define ITERATIONS ( 100 )
#define NOF_PROCESSING_STEPS 9
#define NOF_IMAGES 6


int main(int argc, char** argv)
{
	int height[NOF_IMAGES] = {512, 1024, 2048, 4096, 4096, 8192};
	int width[NOF_IMAGES] = {256, 512, 1024, 2048, 4096, 4096};

	float timing[NOF_PROCESSING_STEPS];
	float avg_time[NOF_PROCESSING_STEPS];
	clock_t start, end;

	CudaGPU device(0);
	CudaBase cub(&device);

	FILE* fid;
	fid = fopen("results.csv", "w");
	fprintf(fid, "Size,Win cal, Win real,Win cplx,Tranpose,Hilbert,FFT(C2C),FFT(R2C),IFFT(C2C),Abs\n");
	//sleep(5);
	printf("Starting simulation of processing steps... \n");
	for(int i = 0; i < NOF_IMAGES; i++)
	{
		printf("Operating on a %d x %d image...",height[i], width[i]);

		fprintf(fid, "%dx%d,",height[i], width[i]);
		cub.setLength(width[i]);
		cub.setHeight(height[i]);
		cub.initDeviceEnv();
		for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
		{
			avg_time[j] = 0;
			timing[j] = 0;
		}
		for(int i = 0; i < ITERATIONS; i++)
		{
			printf("\n\n________________\nRun (%d/%d) \n",i+1,ITERATIONS);
			start = clock();
			cub.setWindow(HAMMING, REAL);
			end = clock();
			timing[0] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.windowReal(cub.getFloatBuffer(),width[i], height[i]);
			end = clock();
			timing[1] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.windowCplx(cub.getComplexBuffer(),width[i], height[i]);
			end = clock();
			timing[2] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.transpose(cub.getComplexBuffer(), cub.getTransposeBuffer(), width[i], height[i]);
			end = clock();
			timing[3] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.hilbertTransform(cub.getFloatBuffer(), cub.getComplexBuffer(), width[i], height[i]);
			end = clock();
			timing[4] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.c2c1dFFT(cub.getComplexBuffer(), width[i], height[i]);
			end = clock();
			timing[5] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.r2c1dFFT(cub.getFloatBuffer(), cub.getComplexBuffer(), width[i], height[i]);
			end = clock();
			timing[6] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.c2c1dIFFT(cub.getComplexBuffer(), width[i], height[i]);
			end = clock();
			timing[7] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

			start = clock();
			cub.absolute(cub.getComplexBuffer(), cub.getFloatBuffer(), width[i], height[i]);
			end = clock();
			timing[8] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;
		}
		for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
		{
			avg_time[j] = timing[j] / ITERATIONS;
			if(j == NOF_PROCESSING_STEPS -1)
				fprintf(fid, "  %.4f \n", avg_time[j]);
			else
				fprintf(fid, "  %.4f,", avg_time[j]);
		}
		cub.freeMemory();
		sleep(1);
	}
	fclose(fid);
}
