#include "CudaGPU.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

CudaGPU::CudaGPU(int devNum)
{
    printf("Starting CUDA device query...\n");
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    checkMemory();
    setDeviceID(devNum);
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
    std::size_t free, total;
    cuMemGetInfo(&free, &total);
    if(size != 0 && print)
        printf("GPU memory (free): %.4f MBytes\n", (float)free/(1024*1024));
    setFreeMemory(free);
    setTotalMemory(total);
    if(free_mem < size)
        return 0;
    return 1;
}
