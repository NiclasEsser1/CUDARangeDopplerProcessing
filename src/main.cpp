#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaAlgorithm.cuh"
#include "CudaTest.h"
#include "Bitmap_IO.h"

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define ITERATIONS ( 10 )
#define NOF_PROCESSING_STEPS 9
#define NOF_IMAGES 7


void testClasses();
void kernelPerformanceTest();

int main(int argc, char** argv)
{
	// testClasses();
	kernelPerformanceTest();
	return 0;
}


void testClasses()
{
	CudaGPU device(0);
	CudaBase cu_base(&device);
	CudaAlgorithm cu_algo(&cu_base);

	// CudaTest<CudaBase> test_base(&device, &cu_base);
	// test_base.testCudaBase(128, 256);

	CudaTest<CudaAlgorithm> test_algo(&device, &cu_algo);
	test_algo.testCudaAlgorithms(512, 512);
}


void kernelPerformanceTest()
{
	int height[NOF_IMAGES] = {512, 1024, 2048, 4096, 4096, 8192, 16384};
	int width[NOF_IMAGES] = {256, 512, 1024, 2048, 4096, 4096, 8192};

	float timing[NOF_PROCESSING_STEPS];
	float avg_time[NOF_PROCESSING_STEPS];
	clock_t start, end;

	CudaGPU device(0);
	CudaBase cu_base(&device);
	CudaAlgorithm cu_algo(&cu_base);

	FILE* fid;
	fid = fopen("results.csv", "w");
	fprintf(fid, "Size,Win cal, Win real,Win cplx,Tranpose,Hilbert,FFT(C2C),FFT(R2C),IFFT(C2C),Abs\n");
	printf("Starting simulation of processing steps... \n");
	for(int k = 0; k < NOF_IMAGES; k++)
	{
		printf("Operating on a %d x %d image...",height[k], width[k]);

		fprintf(fid, "%dx%d,",height[k], width[k]);

		cu_algo.setWidth(width[k]);
		cu_algo.setHeight(height[k]);

		if(cu_algo.initDeviceEnv())
		{
			for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
			{
				avg_time[j] = 0;
				timing[j] = 0;
			}
			for(int i = 0; i < ITERATIONS; i++)
			{
				//printf("\n\n________________\nRun (%d/%d) \n",i+1,ITERATIONS);
				start = clock();
				cu_base.setWindow(cu_algo.getWindowBuffer(), cu_algo.getWindowSize(), HAMMING, REAL);
				end = clock();
				timing[0] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.windowReal(cu_algo.getFloatBuffer(), cu_algo.getWindowBuffer(), width[k], height[k]);
				end = clock();
				timing[1] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.windowCplx(cu_algo.getComplexBuffer(), cu_algo.getWindowBuffer(), width[k], height[k]);
				end = clock();
				timing[2] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.transpose(cu_algo.getComplexBuffer(), width[k], height[k]);
				end = clock();
				timing[3] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.hilbertTransform(cu_algo.getFloatBuffer(), cu_algo.getComplexBuffer(), width[k], height[k]);
				end = clock();
				timing[4] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.c2c1dFFT(cu_algo.getComplexBuffer(), width[k], height[k]);
				end = clock();
				timing[5] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.r2c1dFFT(cu_algo.getFloatBuffer(), cu_algo.getComplexBuffer(), width[k], height[k]);
				end = clock();
				timing[6] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.c2c1dIFFT(cu_algo.getComplexBuffer(), width[k], height[k]);
				end = clock();
				timing[7] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.absolute(cu_algo.getComplexBuffer(), cu_algo.getFloatBuffer(), width[k], height[k]);
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
			if(k != NOF_IMAGES-1)
				cu_algo.freeMemory();
			sleep(1);
		}
	}
	fclose(fid);
}
