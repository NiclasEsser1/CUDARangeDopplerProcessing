#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaAlgorithm.cuh"
#include "CudaTest.h"
#include "Bitmap_IO.h"

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define ITERATIONS ( 100 )
#define NOF_PROCESSING_STEPS 9
#define NOF_IMAGES 8


void testClasses();
void kernelPerformanceTest();

int main(int argc, char** argv)
{
	testClasses();
	return 0;
}


void testClasses()
{
	CudaGPU device(0);
	CudaBase cu_base(&device);
	CudaAlgorithm cu_algo(&cu_base, 512, 512, 1, 3);

	// CudaTest<CudaBase> test_base(&device, &cu_base);
	// test_base.testCudaBase(256, 256);

	CudaTest<CudaAlgorithm> test_algo(&device, &cu_algo);
	test_algo.testCudaAlgorithms();
}
