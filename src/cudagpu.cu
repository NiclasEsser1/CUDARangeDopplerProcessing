#include "cudagpu.cuh"


CudaGPU::CudaGPU(int devNum)
{
    id = devNum;
    printf("Starting CUDA device query...\n");
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s), choosed device %d\n", deviceCount, id);
    }
    (cuMemGetInfo(&free_mem, &total_mem));
}

CudaGPU::~CudaGPU()
{
    cudaDeviceReset();
}

void CudaGPU::setDeviceID(int val)
{
    id = val;
}

int CudaGPU::getDeviceID()
{
    return id;
}

cudaDeviceProp CudaGPU::getProperties()
{
    return prop;
}
int CudaGPU::checkMemory(size_t size, bool print)
{
    if(size != 0 && print)
		printf("GPU free mem: (%.2f/%.2f) MBytes\n", (float)free_mem/(1024*1024), (float)total_mem/(1024*1024));
	if(free_mem < size)
		return 0;
    free_mem = free_mem - size;
	return 1;
}
