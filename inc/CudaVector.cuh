#ifndef CUDAVECTOR_CUH_
#define CUDAVECTOR_CUH_

#include "CudaGPU.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>


template <typename T>
class CudaVector
{
private:
    T* m_bValues;
    std::size_t m_bSize;
    std::size_t eleSize;
public:
    __host__
    void* operator new(std::size_t size)
    {
        CudaVector<T>* object = NULL;
        object = (CudaVector<T> *)malloc(size*sizeof(CudaVector<T>));
        return object;
    }

    __host__
    void operator delete(void* object)
    {
        free(object);
    }

    __host__
    CudaVector(CudaGPU* gpu, std::size_t size = 1)
    {
        m_bSize = size;
        eleSize = sizeof(T);
        if(gpu->checkMemory(size))
        {
            printf("Allocating memory: %.4f MBytes\n", (float)size * sizeof(T)/(1024*1024));
            CUDA_CHECK(cudaMalloc(&m_bValues, m_bSize * sizeof(T)));
            CUDA_CHECK(cudaMemset(m_bValues, 0, m_bSize));
        }
        else
        {
            printf("GPU memory is out of range, could not allocate memory..\n");
            exit(1);
        }
    }

    __host__
    ~CudaVector()
    {
    	if(m_bValues != NULL)
    		CUDA_CHECK(cudaFree(m_bValues));
    }

    __host__
    T* getDevPtr()
    {
        return m_bValues;
    }

    __host__
    std::size_t getSize()
    {
        return m_bSize;
    }

    __host__
    std::size_t geteleSize()
    {
        return eleSize;
    }
};

#endif
