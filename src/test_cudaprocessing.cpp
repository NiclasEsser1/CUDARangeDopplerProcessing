#include "cudagpu.cuh"
#include "cudabase.cuh"
#include "cudaalgorithm.cuh"
#include "cudatest.h"
#include "bitmap_io.h"

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>

#define TEST_CASES 1


void testClasses();

int main(int argc, char** argv)
{
	testClasses();
	return 0;
}


void testClasses()
{
	CudaGPU device(0);


	for(int i = 1; i <= TEST_CASES; i++)
	{
		CudaBase cu_base(&device);
		CudaTest<CudaBase> test_base(&device, &cu_base);
		test_base.testCudaBase(2048, 1024);
		cu_base.~CudaBase();
	}
}
