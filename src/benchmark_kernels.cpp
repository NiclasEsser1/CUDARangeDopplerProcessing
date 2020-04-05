#include "cudaalgorithm.cuh"
#include "signalgenerator.h"
#include "utils.h"

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define ITERATIONS ( 100 )
#define NOF_PROCESSING_STEPS 20
#define NOF_IMAGES 32

using namespace utils;

int main(int argc, char** argv)
{
	int height[NOF_IMAGES];
	int width[NOF_IMAGES];
	for(int i = 0; i < NOF_IMAGES; i++)
	{
		height[i] = 256*(i+1);
		width[i] = height[i];
	}
	int jpeg_size;

	float timing[NOF_PROCESSING_STEPS];
	float avg_time[NOF_PROCESSING_STEPS];
	clock_t start, end;

	processing_conf *conf = nullptr;
	CudaGPU device(0);
	CudaBase cu_base(&device);
	CudaAlgorithm cu_algo(&device, conf);
	SignalGenerator *signal;

	FILE* fid;
	fid = fopen("./results/benchmarks/benchmark_kernels.csv", "w");
	fprintf(fid, "Size,Win cal,win2d cal,Win real,Win cplx,Tranpose (slow),Tranpose (fast),transpose shared, hermetian transpose (slow), hermetian transpose (fast), hermetian transpose shared, fftshift, Abs,Max,Min,minmax, mapcolor, encodebmpjpeg, Hilbert,FFT(C2C),FFT(R2C),IFFT(C2C)\n");
	msg("Starting simulation of processing steps... ");

	for(int k = 0; k < NOF_IMAGES; k++)
	{
		msg("Operating on a "+to_string(height[k])+" x "+to_string(width[k])+" image...");

		fprintf(fid, "%dx%d,",height[k], width[k]);
		cu_algo.setWidth(width[k]);
		cu_algo.setHeight(height[k]);
		uint8_t* jpeg = new uint8_t(width[k]*height[k]);
		if(cu_algo.initProcessingEnv())
		{
			for(int j = 0; j < NOF_PROCESSING_STEPS; j++)
			{
				avg_time[j] = 0;
				timing[j] = 0;
			}

			for(int i = 0; i < ITERATIONS; i++)
			{
				if(!(i%1))
					msg("\n\n________________\nRun ("+to_string(i)+"/"+to_string(ITERATIONS)+")");

				start = clock();
				cu_base.setWindow(cu_algo.getWindowBuffer(), width[k], HAMMING);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[0] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.setWindow(cu_algo.getFloatBuffer(), width[k], HAMMING, height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[1] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.window(cu_algo.getFloatBuffer(), cu_algo.getWindowBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[2] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.window(cu_algo.getComplexBuffer(), cu_algo.getWindowBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[3] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// In place transpose with temporary buffer (slow)
				start = clock();
				cu_base.transpose(cu_algo.getFloatBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[4] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// Without temporary buffer (fast)#
				start = clock();
				cu_base.transpose(cu_algo.getFloatBuffer(), width[k], height[k], cu_algo.getFloatBuffer());
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[5] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// In place transpose with shared
				start = clock();
				cu_base.transposeShared(cu_algo.getFloatBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[6] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// In place hermetian transpose (slow)
				start = clock();
				cu_base.hermitianTranspose(cu_algo.getComplexBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[7] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// // hermetian  transpose (fast)
				start = clock();
				cu_base.hermitianTranspose(cu_algo.getComplexBuffer(), width[k], height[k], cu_algo.getComplexBuffer());
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[8] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// hermetian  transpose shared L2 (very fast)
				start = clock();
				cu_base.hermetianTransposeShared(cu_algo.getComplexBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[9] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// fftshift
				start = clock();
				cu_base.fftshift(cu_algo.getComplexBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[10] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.absolute(cu_algo.getComplexBuffer(), cu_algo.getFloatBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[11] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.max(cu_algo.getFloatBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[12] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.min(cu_algo.getFloatBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[13] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.minMax(cu_algo.getFloatBuffer(), cu_algo.getFloatBuffer(), width[k]*height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[14] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.mapColors(cu_algo.getFloatBuffer(), cu_algo.getCharBuffer(), width[k], height[k], JET);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[15] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				//cu_base.encodeBmpToJpeg(cu_algo.getCharBuffer(), jpeg, &jpeg_size, width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
			 	timing[16] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				// start = clock();
				// if(width[k] != 8192)
				// {
				// 	cu_base.hilbertTransform(cu_algo.getFloatBuffer(), cu_algo.getComplexBuffer(), width[k], height[k]);
				// 	CUDA_CHECK(cudaDeviceSynchronize());
				// }
				// end = clock();
				// timing[17] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.c2c1dFFT(cu_algo.getComplexBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[18] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.r2c1dFFT(cu_algo.getComplexBuffer(), width[k], height[k], cu_algo.getFloatBuffer());
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[19] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

				start = clock();
				cu_base.c2c1dInverseFFT(cu_algo.getComplexBuffer(), width[k], height[k]);
				CUDA_CHECK(cudaDeviceSynchronize());
				end = clock();
				timing[20] += ((float)(end-start) + 1) * 1000 / (float)CLOCKS_PER_SEC;

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
				cu_algo.freeMemory();
				//signal->freeBuffer(signal->getSignal());
			sleep(1);
		}
		else
		{
			k = NOF_IMAGES;
		}
		//printf("________________\n\n\n");
	}
	fclose(fid);
}
