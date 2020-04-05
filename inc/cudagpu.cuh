#ifndef CUDAGPU_H_
#define CUDAGPU_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <curand.h>
#include <libgpujpeg/gpujpeg.h>
#include <nvjpeg.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define CURAND_CALL(ans) { __curandCheckError((ans), __FILE__ ,__LINE__);}
#define CUDA_CHECK(ans) { __cudaCheckError((ans), __FILE__, __LINE__); }


enum { HAMMING = 1, HANN, BARTLETT, BLACKMAN };
enum { RANGE = 1, RANGE_DOPPLER };
enum { JET = 1, VIRIDIS, ACCENT, MAGMA, INFERNO, BLUE};
enum { NONE, BMP24, JPG};
enum { LINEAR, LOG};


class CudaGPU
{
public:
    CudaGPU(int devNum = 0);
    ~CudaGPU();
    void setDeviceID(int val);
    int getDeviceID();
    cudaDeviceProp getProperties();


	int checkMemory(std::size_t size = 0, bool print = false);
	void freedMemory(std::size_t size){free_mem += size;}
	std::size_t getFreeMemory(){checkMemory();return free_mem;};
	std::size_t totalMemory(){checkMemory();return total_mem;}
	std::size_t sharedMemPerBlock(){return prop.sharedMemPerBlock;}
private:
    int id;
    cudaDeviceProp prop;
    std::size_t free_mem;
    std::size_t total_mem;
};


inline void __curandCheckError(curandStatus_t code, const char *file, int line, bool abort = true)
{
	if(code != CURAND_STATUS_SUCCESS)
	{
		std::cout << "Error in file " << file << " at line " << std::to_string(line) << std::endl;
	}
}
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

#endif
