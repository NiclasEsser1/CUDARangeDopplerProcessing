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

#define TEST_CASES 8


void testClasses();

int main(int argc, char** argv)
{
	testClasses();
	return 0;
}


void testClasses()
{
	CudaGPU device(0);


	for(int i = 1; i < TEST_CASES; i++)
	{
		CudaBase cu_base(&device);
		CudaTest<CudaBase> test_base(&device, &cu_base);
		test_base.testCudaBase(16*i, 16*i);
		cu_base.~CudaBase();
	}
	CudaBase cu_base(&device);
	CudaAlgorithm cu_algo(&cu_base, 512, 512, 1, 3);
	CudaTest<CudaAlgorithm> test_algo(&device, &cu_algo);
	test_algo.testCudaAlgorithms();
}
