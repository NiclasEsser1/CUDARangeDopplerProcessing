#ifndef CUDAGPU_H_
#define CUDAGPU_H_

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(ans) { __cudaCheckError((ans), __FILE__, __LINE__); }
inline void __cudaCheckError(cudaError_t code, const char *file, int line, bool abort = true)
{
	cudaError sync = cudaDeviceSynchronize();
	// cudaSuccess async = cudaDeviceASynchronize();
	if(sync != cudaSuccess)
	{
		if (code != cudaSuccess)
		{
			printf("Synchronous Cuda Error: %s %s %d\n",
				cudaGetErrorString(code),
				file,
				line);
		}
	}
	if (code != cudaSuccess)
	{
		printf("Cuda Error: %s %s %d\n",
			cudaGetErrorString(code),
			file,
			line);
	}
}

class CudaGPU
{
public:
    CudaGPU(int devNum = 0);
    ~CudaGPU();
    void setDeviceID(int val);
    int getDeviceID();
    cudaDeviceProp getProperties();
private:
    int id;
    cudaDeviceProp prop;
};

#endif
