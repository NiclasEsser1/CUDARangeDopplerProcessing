#include "main.h"
#include "Stopwatch.h"
#include "SignalGenerator.h"
#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

int main(int argc, char** argv)
{
	float fc = 10.0;
	float fs = 20.0;
	int rec_length = 256;
	int records = 512;
	int channels = 512;

	SignalGenerator sig(fs, fc, rec_length, records, channels);
	CudaGPU device(0);

	// timerclk::precise_stopwatch Stopwatch;
	sig.rectangle();

	CudaBase cub(&device, rec_length, records);

	// sleep(5);
	cub.setProcessingBuffer(sig.getSignal());

	cub.setWindow(HAMMING, REAL, D_GLOBAL);
	// cub.printWindowTaps();
	cub.rangeDopplerProcessing();
}


// float *d;
// float* signal = sig.getSignal();
// float *h = (float*)malloc(rec_length*records*channels*sizeof(float));
// (cudaMalloc((void**)&d, rec_length*records*channels*sizeof(float)));
// (cudaMemcpy((void*)d, (void*)signal, rec_length*records*channels*sizeof(float), cudaMemcpyHostToDevice));
// (cudaMemcpy((void*)h, (void*)d, rec_length*records*channels*sizeof(float), cudaMemcpyDeviceToHost));
// for(int i = 0; i < rec_length*records*channels*sizeof(float); i++)
// 	printf("h[%d] = %f\n", i, h[i]);
//
