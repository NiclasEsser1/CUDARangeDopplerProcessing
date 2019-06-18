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
